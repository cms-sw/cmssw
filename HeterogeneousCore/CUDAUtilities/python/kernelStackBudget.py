"""Estimate the worst-case CUDA local-memory backing store reserved by a cmsRun job.

A CUDA kernel with a large per-thread stack frame makes the driver reserve
cudaLimitStackSize * maxResidentThreads bytes of device memory, where
maxResidentThreads = multiProcessorCount * maxThreadsPerMultiProcessor. The limit must cover
the largest per-thread stack frame among the kernels that launch, so the reservation can
quietly consume a large slice of device memory.

By default the tool works statically: it loads the configuration, resolves each scheduled
module to its plugin library, reads the per-kernel STACK size with cuobjdump, and reports
maxResidentThreads * max(STACK) per device. That is a conservative upper bound over every
kernel in those libraries. The --launched mode runs the job once under a CUPTI logger and
reports only the kernels that actually launched.

Alongside the stack budget it reports each kernel's per-block shared-memory use: the static
amount from cuobjdump and, in --launched mode, the dynamic amount requested at launch (from
CUPTI) and their total. That helps decide whether register/local spills could be steered into
the on-chip L1/shared memory instead of global memory.

If the target device has no compatible embedded SASS arch (it would JIT-compile the PTX at
runtime), the tool exits with an error rather than report figures that would not describe what
actually runs; arch compatibility follows the CUDA rules for base, family ('f') and accelerated
('a') targets.
"""

import concurrent.futures
import os
import re
import shutil
import struct
import subprocess
import sys
from collections import defaultdict

from FWCore.ParameterSet.processFromFile import processFromFile

_ALPAKA_SUFFIX = "@alpaka"

# cuobjdump --dump-resource-usage parsing; the arch carries the CUDA 12.9+ variant suffix, e.g.
# "arch = sm_90a" (accelerated) or "arch = sm_100f" (family-specific)
_ARCH_RE = re.compile(r"arch\s*=\s*(sm_\d+[af]?)")
_FUNC_RE = re.compile(r"Function\s+(\S+):")
_USAGE_RE = re.compile(r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)")

# cudaComputeCapabilities --verbose parsing; a device line looks like "   0     9.0    NVIDIA H100 NVL"
_DEVICE_RE = re.compile(r"^\s*(\d+)\s+(\d+\.\d+)\s+(.+?)(?:\s+\(unsupported\))?\s*$")
_MAXRES_RE = re.compile(r"max resident threads:\s*(\d+)")


def find_tool(name):
    """Locate a CUDA/CMSSW tool on $PATH (cmsenv puts the CUDA external's bin/ on it)."""
    return shutil.which(name)


def lib_directories():
    """CMSSW plugin-library directories to search, developer area first, then the release base(s).

    Ordering matters: locally rebuilt plugins in the developer area must shadow the release, so
    the developer-area directories have to come first.

    The directories are read from LD_LIBRARY_PATH, which scram (cmsenv) fills with every plugin
    directory in the right precedence order (local, then cvmfs).
    """
    arch = os.environ.get("SCRAM_ARCH", "")
    dirs = []

    def add(path):
        if path and os.path.isdir(path) and path not in dirs:
            dirs.append(path)

    def is_plugin_dir(path):
        parts = path.rstrip("/").split(os.sep)
        return (len(parts) >= 2 and parts[-2:] == ["lib", arch]) or \
               (len(parts) >= 3 and parts[-3:-1] == ["lib", arch])

    for path in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if is_plugin_dir(path):
            add(path)

    return dirs


def parse_plugin_cache(lib_dirs):
    """Return a {plugin_name: shared_object_basename} map from the .edmplugincache files.

    Cache lines look like "<library.so> <pluginName> <category>". Developer-area caches are
    read last so locally rebuilt plugins win over the base release.
    """
    mapping = {}
    for directory in reversed(lib_dirs):  # base release first, local area overwrites
        cache = os.path.join(directory, ".edmplugincache")
        try:
            with open(cache) as handle:
                for line in handle:
                    fields = line.split()
                    if len(fields) >= 2:
                        mapping[fields[1]] = fields[0]
        except OSError:
            continue
    return mapping


def resolve_library_path(basename, lib_dirs):
    """Resolve a shared-object basename to a full path (developer area first)."""
    for directory in lib_dirs:
        candidate = os.path.join(directory, basename)
        if os.path.exists(candidate):
            return candidate
    return None


def _module_backend(module, default_backend):
    """Return the Alpaka backend an @alpaka module resolves to.

    A per-module 'alpaka.backend' set explicitly in the configuration wins; otherwise the
    process-wide default_backend applies (see config_backend).
    """
    alpaka = getattr(module, "alpaka", None)
    if alpaka is not None and hasattr(alpaka, "backend"):
        backend = alpaka.backend.value()
        if backend:
            return backend
    return default_backend


def plugin_name_for(cpp_type, module, default_backend):
    """Map a module C++ type to its registered plugin name.

    "Foo@alpaka" becomes "alpaka_<backend>::Foo", mirroring ModuleTypeResolverAlpaka in
    HeterogeneousCore/AlpakaCore. Any other type is its own plugin name.
    """
    if cpp_type.endswith(_ALPAKA_SUFFIX):
        base = cpp_type[: -len(_ALPAKA_SUFFIX)]
        return "alpaka_{}::{}".format(_module_backend(module, default_backend), base)
    return cpp_type


# accelerator name (process.options.accelerators) -> Alpaka backend, in the priority order
# ModuleTypeResolverAlpaka uses when picking the default backend
_ACCELERATOR_BACKENDS = (
    ("gpu-nvidia", "cuda_async"),
    ("gpu-amd", "rocm_async"),
    ("cpu", "serial_sync"),
)
# the Alpaka backend that emits CUDA device code: the only backend this tool can analyse, the
# natural default, and what '*' (all accelerators, nvidia first) resolves to here
_CUDA_BACKEND = "cuda_async"


def config_backend(process):
    """Return the Alpaka backend @alpaka modules resolve to, honouring the configuration.

    Mirrors HeterogeneousCore/AlpakaCore ModuleTypeResolverAlpaka: an explicit
    ProcessAcceleratorAlpaka.setBackend(...) wins; otherwise the backend follows
    process.options.accelerators (the first of gpu-nvidia -> cuda_async, gpu-amd -> rocm_async,
    cpu -> serial_sync that the requested accelerators allow). '*' (the default) allows all, which
    for this tool means cuda_async. Anything that cannot be read falls back to cuda_async.
    """
    import fnmatch

    # an explicit ProcessAcceleratorAlpaka.setBackend(...) takes precedence over the accelerators
    try:
        accelerators = process.processAccelerators_()
    except (AttributeError, TypeError):
        accelerators = {}
    for accelerator in accelerators.values():
        if type(accelerator).__name__ == "ProcessAcceleratorAlpaka":
            backend = getattr(accelerator, "_backend", None)
            if backend:
                return backend

    # otherwise follow process.options.accelerators (patterns such as 'gpu-*' are allowed)
    try:
        selected = list(process.options.accelerators)
    except (AttributeError, TypeError):
        selected = []
    if selected and "*" not in selected:
        for name, backend in _ACCELERATOR_BACKENDS:
            if any(fnmatch.fnmatch(name, pattern) for pattern in selected):
                return backend
    return _CUDA_BACKEND


def scheduled_modules(process):
    """Return [(label, cpp_type, module)] for modules that would run for this configuration.

    Uses the explicit cms.Schedule when present (it already pulls in associated Tasks),
    otherwise the union of all Paths and EndPaths. Alpaka ESProducers are added always because
    they are loaded on demand rather than scheduled on a path.
    """
    labels = set()
    schedule = process.schedule_() if hasattr(process, "schedule_") else None
    if schedule is not None:
        labels |= set(schedule.moduleNames())
    else:
        for container in (process.paths_(), process.endpaths_()):
            for path in container.values():
                labels |= set(path.moduleNames())

    lookup = {}
    for getter in ("producers_", "filters_", "analyzers_", "outputModules_"):
        lookup.update(getattr(process, getter)())

    result = []
    for label in sorted(labels):
        module = lookup.get(label)
        if module is not None:
            result.append((label, module.type_(), module))

    for label, module in process.es_producers_().items():
        cpp_type = module.type_()
        if cpp_type.endswith(_ALPAKA_SUFFIX):
            result.append((label, cpp_type, module))

    return result


def loaded_cuda_libraries(process, default_backend, plugin_cache, lib_dirs):
    """Return ({library_path: sorted labels}, unresolved) for the configuration's CUDA libraries.

    A library is kept only if it embeds device code, checked with has_device_code.
    """
    libraries = defaultdict(set)
    unresolved = []
    for label, cpp_type, module in scheduled_modules(process):
        plugin = plugin_name_for(cpp_type, module, default_backend)
        basename = plugin_cache.get(plugin)
        if basename is None:
            unresolved.append((label, plugin))
            continue
        path = resolve_library_path(basename, lib_dirs)
        if path is None:
            unresolved.append((label, basename))
            continue
        if not has_device_code(path):
            continue
        libraries[path].add(label)
    return {path: sorted(labels) for path, labels in libraries.items()}, unresolved


def parse_resource_usage(text):
    """Parse cuobjdump --dump-resource-usage output into per-kernel records."""
    records = []
    arch = None
    kernel = None
    for line in text.splitlines():
        match = _ARCH_RE.search(line)
        if match:
            arch, kernel = match.group(1), None
            continue
        match = _FUNC_RE.search(line)
        if match:
            kernel = match.group(1)
            continue
        match = _USAGE_RE.search(line)
        # require an arch context too, so a stray usage block never yields an arch=None record
        if match and kernel is not None and arch is not None:
            reg, stack, shared, local = (int(value) for value in match.groups())
            records.append(
                {"arch": arch, "kernel": kernel, "reg": reg, "stack": stack, "shared": shared, "local": local}
            )
            kernel = None
    return records


def kernel_resource_usage(library_path, cuobjdump):
    """Run cuobjdump on a library and return its per-kernel resource records (or [])."""
    try:
        completed = subprocess.run(
            [cuobjdump, "--dump-resource-usage", library_path],
            capture_output=True,
            text=True,
            errors="replace",
        )
    except OSError:
        return []
    # cuobjdump exits non-zero for host-only libraries; an empty parse is the natural result
    return parse_resource_usage(completed.stdout)


def has_device_code(path):
    """True if the ELF64 shared object embeds a CUDA fatbin section."""
    try:
        with open(path, "rb") as handle:
            header = handle.read(64)
            if header[:4] != b"\x7fELF" or header[4] != 2:  # ELF64 only (the build platform)
                return False
            endian = "<" if header[5] == 1 else ">"
            sh_offset = struct.unpack_from(endian + "Q", header, 40)[0]
            sh_entsize = struct.unpack_from(endian + "H", header, 58)[0]
            sh_count = struct.unpack_from(endian + "H", header, 60)[0]
            str_index = struct.unpack_from(endian + "H", header, 62)[0]
            if not sh_offset or not sh_entsize:
                return False
            # extended numbering keeps the real count and string-table index in section 0
            if sh_count == 0 or str_index == 0xFFFF:
                handle.seek(sh_offset)
                first = handle.read(sh_entsize)
                if sh_count == 0:
                    sh_count = struct.unpack_from(endian + "Q", first, 32)[0]  # sh_size
                if str_index == 0xFFFF:
                    str_index = struct.unpack_from(endian + "I", first, 40)[0]  # sh_link
            if str_index >= sh_count:
                return False
            handle.seek(sh_offset)
            sections = handle.read(sh_entsize * sh_count)
            base = str_index * sh_entsize
            str_offset = struct.unpack_from(endian + "Q", sections, base + 24)[0]
            str_size = struct.unpack_from(endian + "Q", sections, base + 32)[0]
            handle.seek(str_offset)
            strtab = handle.read(str_size)
            return b".nv_fatbin" in strtab or b"__nv_relfatbin" in strtab
    except (OSError, struct.error, IndexError):
        return False


def demangle(names):
    """Return a {mangled: demangled} map using a C++ demangler, falling back to identity.

    Prefers llvm-cxxfilt: GNU c++filt gives up on very long mangled names (deep Alpaka
    template instantiations exceed its demangler limits) and returns them unchanged.
    """
    names = list(names)
    cxxfilt = shutil.which("llvm-cxxfilt") or shutil.which("c++filt")
    if not cxxfilt or not names:
        return {name: name for name in names}
    try:
        completed = subprocess.run(
            [cxxfilt], input="\n".join(names), capture_output=True, text=True, errors="replace"
        )
        demangled = completed.stdout.splitlines()
        if len(demangled) == len(names):
            return dict(zip(names, demangled))
    except OSError:
        pass
    return {name: name for name in names}


def short_kernel_name(name):
    """Reduce an Alpaka kernel name to the meaningful inner kernel type.

    Alpaka wraps every kernel as alpaka::detail::gpuKernel<REAL_KERNEL, Acc, ...>(...); the
    first template argument is the kernel that matters. Names without that wrapper (e.g. plain
    CUDA kernels) are returned unchanged.
    """
    marker = "gpuKernel<"
    start = name.find(marker)
    if start < 0:
        return name
    start += len(marker)
    depth = 1
    index = start
    while index < len(name) and depth > 0:
        char = name[index]
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
            if depth == 0:
                break
        elif char == "," and depth == 1:
            break
        index += 1
    return name[start:index].strip()


def compute_capability_to_sm(compute_capability):
    """Convert a compute capability ('9.0', '90', 'sm_90' or 'sm_9.0') to a cuobjdump arch.

    Always returns 'sm_<digits>' so callers can int() the numeric part; a dotted form
    ('9.0' or 'sm_9.0') has its separator removed rather than passed through verbatim.
    """
    text = str(compute_capability)
    if text.startswith("sm_"):
        text = text[len("sm_"):]
    if "." in text:
        major, minor = text.split(".", 1)
        return "sm_{}{}".format(major, minor)
    return "sm_{}".format(text)


def parse_device_info(text):
    """Parse cudaComputeCapabilities --verbose output into device descriptors."""
    devices = []
    current = None
    for line in text.splitlines():
        match = _DEVICE_RE.match(line)
        if match:
            current = {
                "index": int(match.group(1)),
                "compute_capability": match.group(2),
                "name": match.group(3).strip(),
                "max_resident_threads": None,
            }
            devices.append(current)
            continue
        if current is not None:
            match = _MAXRES_RE.search(line)
            if match:
                current["max_resident_threads"] = int(match.group(1))
    return devices


def detect_devices():
    """Detect local devices by running cudaComputeCapabilities --verbose (or [])."""
    tool = find_tool("cudaComputeCapabilities")
    if not tool:
        return []
    try:
        completed = subprocess.run([tool, "--verbose"], capture_output=True, text=True, errors="replace")
    except OSError:
        return []
    if completed.returncode != 0:
        # surface the real CUDA error (driver/permission/no-device) instead of letting the
        # empty parse masquerade as "no devices detected"
        if completed.stderr:
            sys.stderr.write(completed.stderr)
        return []
    return parse_device_info(completed.stdout)


def _parse_arch(sm):
    """Split an arch string into (compute-capability number, variant suffix).

    'sm_75' -> (75, ''), 'sm_100' -> (100, ''), 'sm_100f' -> (100, 'f'), 'sm_90a' -> (90, 'a').
    The 'f' (family-specific) and 'a' (architecture-specific, "accelerated") suffixes are the CUDA
    12.9+ arch-conditional variants that cuobjdump reports in its "arch = ..." line.
    """
    body = sm.split("_", 1)[1]
    suffix = ""
    if body and body[-1] in "af":
        suffix, body = body[-1], body[:-1]
    return int(body), suffix


# preference when several compatible arch variants are embedded for the same function: accelerated
# ('a') over family ('f') over base, then the higher compute capability
_ARCH_VARIANT_PRIORITY = {"a": 2, "f": 1, "": 0}


def _arch_preference(arch):
    """Sort key for the arch the driver would prefer: variant first (a > f > base), then CC."""
    number, suffix = _parse_arch(arch)
    return (_ARCH_VARIANT_PRIORITY[suffix], number)


def _sm_major_minor(sm):
    """Split an arch string into (major, minor), ignoring any variant suffix.

    Compute capabilities have a single-digit minor revision, so the last digit of the number is the
    minor and everything before it is the major (two digits from Blackwell/sm_10x onwards).
    """
    number, _ = _parse_arch(sm)
    return number // 10, number % 10


# major compute-capability generations whose base (no-suffix) SASS is forward-compatible beyond the
# usual same-major rule. Blackwell datacenter (sm_10x) and consumer (sm_12x) share this: base sm_100
# also runs on sm_120, whereas sm_100f (family-specific) and sm_100a (accelerated) do not.
_BASE_COMPATIBLE_GENERATIONS = (frozenset({10, 12}),)


def _arch_runs_on(arch, device_number):
    """True if SASS compiled for `arch` runs on a device of compute capability `device_number`.

    `arch` is a cuobjdump arch string, optionally with a variant suffix; `device_number` is the
    device's plain compute capability (e.g. 120 for sm_120). CUDA binary (SASS) compatibility:
      * accelerated ('a', e.g. sm_100a): only the exact same compute capability, nothing else.
      * family ('f', e.g. sm_100f): same major generation, forward across minor revisions.
      * base (no suffix): same major generation forward, plus the cross-generation families NVIDIA
        declares forward-compatible (Blackwell base sm_10x also runs on sm_12x).
    SASS is never backward compatible: a device never runs code built for a higher capability, and
    (for example) a Turing sm_75 device runs Volta sm_70 code but not Pascal sm_6x code.
    """
    number, suffix = _parse_arch(arch)
    if suffix == "a":
        return device_number == number
    if device_number < number:
        return False
    major, device_major = number // 10, device_number // 10
    if device_major == major:
        return True
    if suffix == "f":
        return False
    return any(major in family and device_major in family for family in _BASE_COMPATIBLE_GENERATIONS)


def select_arch(target_sm, available_sms):
    """Pick the embedded SASS arch that will run on a device of compute capability `target_sm`.

    Returns (chosen_sm, fell_back). chosen_sm is None when no embedded arch runs on the device --
    the driver would JIT-compile the embedded PTX, whose per-kernel resource usage cuobjdump cannot
    report (cudaKernelStackBudget treats that as a fatal error). fell_back is True when the chosen
    arch is not the device's own compute capability (a compatible lower/other arch is used instead).

    Compatibility follows _arch_runs_on: exact match for 'a' (accelerated) variants, same major
    generation forward for 'f' (family) variants, and that plus the declared cross-generation
    families for base arches. Among the compatible arches the highest compute capability is chosen.
    PTX JIT resource usage is not known untile runtime: sm incompatibility is treated as an error.
    """
    device_number, _ = _parse_arch(target_sm)
    compatible = [sm for sm in available_sms if _arch_runs_on(sm, device_number)]
    if not compatible:
        return None, False
    chosen = max(compatible, key=lambda sm: _parse_arch(sm)[0])
    return chosen, _parse_arch(chosen)[0] != device_number


def _available_arches(records):
    """Sorted embedded SASS arches, preferring the configuration's own libraries.

    Run-loaded libraries (config=False) are ignored while the configuration's libraries provide any
    arch, so an extra library that happens to embed a different arch cannot skew the selection.
    """
    config_records = [record for record in records if record.get("config", True)]
    pool = config_records if config_records else records
    return sorted({record["arch"] for record in pool}, key=lambda sm: _parse_arch(sm)[0])


def _device_records(records, device_number):
    """Per (kernel, library), the record for the arch the driver would run on the device.

    A function is embedded once per arch; among the arches that run on the device the most specific
    variant is preferred in the following order:
    - accelerated ('a') (e.g. sm_100a)
    - family ('f') (e.g. sm_100f)
    - base (e.g. sm_100)
    - if the code was not compiled for base, take the highest compatible compute capability
    Functions with no device-compatible arch are dropped.
    Selecting per function (rather than filtering on one 'chosen' arch string) keeps arch-variant
    functions that an exact arch-string match would otherwise miss.
    """
    best = {}
    for record in records:
        if not _arch_runs_on(record["arch"], device_number):
            continue
        key = (record["kernel"], record["library"])
        current = best.get(key)
        if current is None or _arch_preference(record["arch"]) > _arch_preference(current["arch"]):
            best[key] = record
    return list(best.values())


def _scan_libraries(paths, cuobjdump, from_config):
    """Run cuobjdump on each library concurrently and return tagged per-kernel records."""
    paths = list(paths)
    if not paths:
        return []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(paths))) as pool:
        per_library = list(pool.map(lambda path: kernel_resource_usage(path, cuobjdump), paths))
    records = []
    for path, found in zip(paths, per_library):
        basename = os.path.basename(path)
        for record in found:
            record["library"] = basename
            record["config"] = from_config
            records.append(record)
    return records


def _human_bytes(value):
    return "{:.1f} MiB ({} B)".format(value / (1024 * 1024), value)


def _worst(records, key):
    return max(records, key=lambda record: record.get(key, 0)) if records else None


def _modules_suffix(kernel, launched_modules):
    """' [moduleA, moduleB]' attribution suffix, or '' when no attribution is available."""
    if launched_modules is None:
        return ""
    modules = launched_modules.get(kernel)
    return "  [{}]".format(", ".join(modules)) if modules else "  [unattributed]"


def _emit_reservation(out, label, worst, max_resident, launched_modules=None, launched=False):
    """Write a 'reservation = threads x stack' block for the worst kernel (or a zero line).

    The block also reports that kernel's per-block shared-memory use: the static amount (from
    cuobjdump) and, when launched is set, the dynamic amount captured at launch and their total.
    """
    if worst is None or worst["stack"] == 0:
        out.write("  {}: {}\n".format(label, _human_bytes(0)))
        return
    out.write("  {}: {} = {} threads x {} B\n".format(
        label, _human_bytes(max_resident * worst["stack"]), max_resident, worst["stack"]))
    out.write("      largest stack frame {} B in {}\n".format(worst["stack"], worst["library"]))
    out.write("      kernel: {}{}\n".format(worst["short"], _modules_suffix(worst["kernel"], launched_modules)))
    static_shared = worst.get("shared", 0)
    if launched:
        dynamic_shared = worst.get("dyn_shared", 0)
        out.write("      shared memory: {} B total = {} B static + {} B dynamic\n".format(
            static_shared + dynamic_shared, static_shared, dynamic_shared))
    else:
        out.write("      shared memory: {} B static\n".format(static_shared))


def _report(out, records, library_labels, devices, unresolved, top, verbose,
            launched_names=None, unmatched_display=None, launched_modules=None):
    """Write the human-readable budget report to out.

    With launched_names (a set of mangled kernel names captured from a real run), the headline
    budget covers only those kernels and the configuration's all-kernels figure is shown as a
    static upper bound; otherwise only the static figure is reported. Records tagged
    config=False come from libraries the run loaded that are not configuration plugins; they
    locate launched kernels and never feed the static bound.
    """
    nlibraries = len(library_labels)
    nmodules = len(set().union(*library_labels.values())) if library_labels else 0
    # the device arch is chosen from the configuration's own libraries when available, so the static
    # bound is never blanked out by an extra run-loaded library that happens to add an arch
    available_sms = _available_arches(records)
    record_names = {record["kernel"] for record in records}
    extra_libraries = {record["library"] for record in records if not record.get("config", True)}

    out.write("CUDA kernel stack-frame memory budget\n")
    out.write("  scheduled modules using CUDA libraries: {}\n".format(nmodules))
    out.write("  CUDA libraries from the configuration:  {}\n".format(nlibraries))
    if launched_names is not None:
        out.write("  extra libraries scanned from the run:   {}\n".format(len(extra_libraries)))
    out.write("  kernels inspected:                      {}\n".format(len(records)))
    if launched_names is not None:
        matched = record_names & launched_names
        out.write("  kernels actually launched (CUPTI):      {} ({} matched to stack data)\n".format(
            len(launched_names), len(matched)))
    if not records:
        out.write("\n  no CUDA kernels found for this configuration: reservation is 0 B.\n")

    for device in devices:
        cc = device["compute_capability"]
        max_resident = device["max_resident_threads"]
        target_sm = compute_capability_to_sm(cc)
        name = device.get("name", "")
        index = device.get("index")
        prefix = "device {}: ".format(index) if index is not None else ""
        out.write("\n{}{} (compute capability {}, {})\n".format(prefix, name, cc, target_sm))

        if max_resident is None:
            out.write("  max resident threads unknown: pass --max-resident-threads.\n")
            continue
        out.write("  max resident threads: {}\n".format(max_resident))

        if not records:
            out.write("  worst-case reservation: {}\n".format(_human_bytes(0)))
            continue

        # main() already rejects a device with no compatible arch (it would JIT from PTX); this is a
        # defensive guard for direct callers of _report
        chosen_sm, fell_back = select_arch(target_sm, available_sms)
        if chosen_sm is None:
            out.write("  no embedded arch runs on {} (have {}); would JIT from PTX, cannot estimate.\n".format(
                target_sm, ", ".join(available_sms)))
            continue
        if fell_back:
            out.write("  WARNING: {} not embedded; using compatible arch {}.\n".format(target_sm, chosen_sm))

        arch_records = _device_records(records, _parse_arch(target_sm)[0])

        if launched_names is None:
            active = arch_records
            _emit_reservation(out, "worst-case reservation", _worst(active, "stack"), max_resident)
        else:
            active = [record for record in arch_records if record["kernel"] in launched_names]
            _emit_reservation(out, "reservation (launched kernels)", _worst(active, "stack"), max_resident,
                              launched_modules, launched=True)
            # the static upper bound stays over the configuration's CUDA libraries only
            static_active = [record for record in arch_records if record.get("config", True)]
            static_worst = _worst(static_active, "stack")
            static_bytes = max_resident * static_worst["stack"] if static_worst else 0
            out.write("  static upper bound (config's CUDA libraries): {}\n".format(_human_bytes(static_bytes)))

        worst_local = _worst(active, "local")
        if worst_local is not None and worst_local["local"] > 0:
            out.write("  (largest local-memory/spill per thread: {} B in {})\n".format(
                worst_local["local"], worst_local["library"]))

        if verbose:
            spilling = sorted(
                (record for record in active if record["stack"] > 0),
                key=lambda record: record["stack"],
                reverse=True,
            )
            out.write("  spilling kernels ({}):\n".format(len(spilling)))
            shown = spilling if top <= 0 else spilling[:top]
            for record in shown:
                # SHARED is the static per-block shared memory; DYN (launched runs only) is the
                # dynamic amount requested at launch, so SHARED + DYN is the per-block total
                dyn_col = "  DYN {:>6} B".format(record.get("dyn_shared", 0)) if launched_names is not None else ""
                out.write("    STACK {:>6} B  LOCAL {:>6} B  SHARED {:>6} B{}  REG {:>3}  {}{}\n".format(
                    record["stack"], record["local"], record.get("shared", 0), dyn_col,
                    record["reg"], record["short"], _modules_suffix(record["kernel"], launched_modules)))
            hidden = spilling[len(shown):]
            if hidden:
                lo, hi = hidden[-1]["stack"], hidden[0]["stack"]
                out.write("    ({} more kernels with STACK {}-{} B not shown; pass --top 0 to list all)\n".format(
                    len(hidden), lo, hi))

    if verbose and library_labels:
        out.write("\nCUDA libraries from the configuration:\n")
        for path in sorted(library_labels):
            out.write("  {}\n".format(os.path.basename(path)))
            out.write("    used by: {}\n".format(", ".join(library_labels[path])))
        if extra_libraries:
            out.write("additional libraries scanned from the run:\n")
            for library in sorted(extra_libraries):
                out.write("  {}\n".format(library))

    if launched_names is not None and unmatched_display:
        out.write("\nNOTE: {} launched kernel(s) were not found in any scanned library, so their\n"
                  "      stack frames are not included (typically statically linked into cmsRun or\n"
                  "      a dependency without a fatbin):\n".format(len(unmatched_display)))
        shown = unmatched_display if top <= 0 else unmatched_display[:top]
        for kernel in shown:
            out.write("  {}\n".format(kernel))
        if len(unmatched_display) > len(shown):
            out.write("  (... {} more; pass --top 0 to list all)\n".format(len(unmatched_display) - len(shown)))

    if unresolved:
        out.write("\nWARNING: {} scheduled module(s) could not be resolved to a library:\n".format(len(unresolved)))
        for label, plugin in unresolved:
            out.write("  {} -> {}\n".format(label, plugin))


def _iter_file_lines(path):
    """Yield the lines of path; yield nothing if it cannot be opened."""
    try:
        handle = open(path)
    except OSError:
        return
    with handle:
        for line in handle:
            yield line


def _read_lines(path):
    """Read a file into the set of its non-empty, stripped lines."""
    return {stripped for line in _iter_file_lines(path) if (stripped := line.strip())}


def _read_launched(path):
    """Read the CuptiKernelLoggerService kernel log.

    Lines are "<mangled kernel>\\t<module label>\\t<max dynamic shared bytes>" (the module may be
    empty).
    Returns the set of launched kernel names, a {kernel: sorted module labels} attribution map,
    and a {kernel: max dynamic shared bytes} map (only kernels seen with a non-zero amount).
    """
    names = set()
    modules = defaultdict(set)
    dynamic_shared = {}
    for line in _iter_file_lines(path):
        fields = line.rstrip("\n").split("\t")
        kernel = fields[0].strip()
        if not kernel:
            continue
        names.add(kernel)
        module = fields[1].strip() if len(fields) > 1 else ""
        if module:
            modules[kernel].add(module)
        if len(fields) > 2:
            try:
                shared = int(fields[2].strip() or "0")
            except ValueError:
                shared = 0
            if shared > dynamic_shared.get(kernel, 0):
                dynamic_shared[kernel] = shared
    return names, {kernel: sorted(labels) for kernel, labels in modules.items()}, dynamic_shared


def capture_launched_kernels(config, config_args):
    """Run cmsRun once with CuptiKernelLoggerService added to the configuration.

    Returns (launched kernel names, {kernel: modules}, {kernel: max dynamic shared bytes},
    loaded library paths, returncode). cmsRun's own output streams to the terminal so the run is
    visible.
    """
    import tempfile

    cmsrun = shutil.which("cmsRun")
    if not cmsrun:
        raise RuntimeError("cmsRun not found on PATH (set up the CMSSW environment)")

    kernel_handle, kernel_log = tempfile.mkstemp(prefix="kernel-log-", suffix=".txt")
    os.close(kernel_handle)
    library_handle, library_log = tempfile.mkstemp(prefix="kernel-libs-", suffix=".txt")
    os.close(library_handle)
    wrapper_handle, wrapper = tempfile.mkstemp(prefix="kernel-budget-cfg-", suffix=".py")
    os.close(wrapper_handle)
    # run the user configuration unchanged, then attach the logging service (mirroring
    # processFromFile's sys.path / __file__ handling so the config resolves the same way)
    config = os.path.abspath(config)
    with open(wrapper, "w") as out:
        out.write(
            "import sys, FWCore.ParameterSet.Config as cms\n"
            "sys.path.insert(0, {dir!r})\n"
            "__file__ = {config!r}\n"
            "exec(compile(open({config!r}).read(), {config!r}, 'exec'))\n"
            "process.add_(cms.Service('CuptiKernelLoggerService',\n"
            "    kernelLog=cms.untracked.string({kernel!r}),\n"
            "    libraryLog=cms.untracked.string({library!r})))\n"
            # subscribe the service in the MessageLogger
            "if not hasattr(process, 'MessageLogger'):\n"
            "    process.load('FWCore.MessageService.MessageLogger_cfi')\n"
            "if not hasattr(process.MessageLogger, 'CuptiKernelLoggerService'):\n"
            "    process.MessageLogger.CuptiKernelLoggerService = cms.untracked.PSet()\n".format(
                dir=os.path.dirname(config), config=config, kernel=kernel_log, library=library_log))
    try:
        completed = subprocess.run([cmsrun, wrapper] + list(config_args))
        names, modules, dynamic_shared = _read_launched(kernel_log)
        return names, modules, dynamic_shared, _read_lines(library_log), completed.returncode
    finally:
        for path in (kernel_log, library_log, wrapper):
            try:
                os.remove(path)
            except OSError:
                pass


def _value_option_strings(parser):
    """Return the set of option strings (e.g. '--top') that consume a following value.

    Derived from the parser itself so _split_cli stays in sync with the declared options: a
    new value-taking flag is recognised automatically, with no second list to update.
    """
    options = set()
    for action in parser._actions:
        if action.option_strings and action.nargs != 0:
            options.update(action.option_strings)
    return options


def _split_cli(argv, value_options=None):
    """Split argv into (tool options, config path, config arguments).

    The first bare token (not an option and not an option's value) is the configuration;
    everything after it is forwarded to the configuration verbatim. This keeps the tool's own
    flags order-independent while letting the config keep flags of the same name. value_options
    is the set of flags that take a separate value (see _value_option_strings); it defaults to
    the tool's own parser so callers and tests need not recompute it.
    """
    if value_options is None:
        value_options = _value_option_strings(_build_parser())
    tool_args = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("-"):
            return tool_args, token, argv[index + 1:]
        tool_args.append(token)
        if token in value_options and index + 1 < len(argv):
            tool_args.append(argv[index + 1])
            index += 2
        else:
            index += 1
    return tool_args, None, []


def _build_parser():
    """Build the argument parser. allow_abbrev is off so abbreviated flags (e.g. --comp) are a
    clean error rather than being silently treated as the CONFIG path by _split_cli."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="cudaKernelStackBudget",
        allow_abbrev=False,
        usage="%(prog)s [options] CONFIG.py [config args...]",
        description="Estimate the worst-case CUDA local-memory backing store reserved by a "
                    "cmsRun configuration, from the largest per-thread kernel stack frame "
                    "(known at compile time) across the plugin libraries the config loads. "
                    "The result is a conservative upper bound (max stack over all kernels in "
                    "the loaded CUDA libraries, not only the kernels each module launches). "
                    "Tool options must precede CONFIG.py; anything after it is forwarded to "
                    "the configuration.",
    )
    parser.add_argument("config", help="cmsRun Python configuration file")
    parser.add_argument("--compute-capability", dest="compute_capability",
                        help="target compute capability, e.g. 9.0 or 90 (default: auto-detect devices)")
    parser.add_argument("--max-resident-threads", dest="max_resident_threads", type=int,
                        help="multiProcessorCount * maxThreadsPerMultiProcessor for the target device")
    parser.add_argument("--launched", action="store_true",
                        help="run cmsRun once with the CUPTI logger service and report only the "
                             "kernels actually launched, with the module that launched each "
                             "(in addition to the static upper bound)")
    parser.add_argument("--top", type=int, default=25,
                        help="cap the verbose spilling-kernel listing at N rows (0 = all; default 25)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print the per-kernel (stack, local and shared memory) and per-library breakdown")
    return parser


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = _build_parser()
    tool_args, config, config_args = _split_cli(argv, _value_option_strings(parser))
    args = parser.parse_args(tool_args + ([config] if config is not None else []))

    if args.compute_capability is not None:
        # validate through the same normalisation the report uses (compute_capability_to_sm)
        try:
            _sm_major_minor(compute_capability_to_sm(args.compute_capability))
        except (ValueError, IndexError):
            parser.error("invalid --compute-capability {!r}; expected e.g. 9.0 or 90".format(args.compute_capability))

    cuobjdump = find_tool("cuobjdump")
    if not cuobjdump:
        parser.error("cuobjdump not found (set up the CUDA environment, e.g. cmsenv)")

    # pass a list (never None) so processFromFile resets the config's sys.argv; otherwise the
    # config would parse this tool's argv
    try:
        process = processFromFile(config, config_args)
    except SystemExit as error:
        parser.error("configuration {} exited during loading (code {})".format(config, error.code))
    except Exception as error:
        parser.error("could not load configuration {}: {}".format(config, error))
    lib_dirs = lib_directories()
    plugin_cache = parse_plugin_cache(lib_dirs)
    # the backend @alpaka modules resolve to comes from the configuration (setBackend / accelerators)
    default_backend = config_backend(process)
    if default_backend != _CUDA_BACKEND:
        # exit from non cuda_async configs
        parser.error(
            "the configuration resolves Alpaka modules to the '{}' backend, not the CUDA backend "
            "'{}'."
            "\nThis job would not launch CUDA kernels, so there is no device-memory budget to "
            "estimate."
            "\nConfigure a CUDA GPU, e.g. process.options.accelerators = ['gpu-nvidia'] or "
            "process.ProcessAcceleratorAlpaka.setBackend('{}').".format(
                default_backend, _CUDA_BACKEND, _CUDA_BACKEND))
    library_labels, unresolved = loaded_cuda_libraries(process, default_backend, plugin_cache, lib_dirs)

    # static analyzer: every kernel in the configuration's CUDA libraries
    records = _scan_libraries(library_labels, cuobjdump, True)

    # resolve the target device(s) and check arch compatibility up front, before the (possibly slow)
    # --launched cmsRun. A device with no compatible embedded SASS arch would JIT-compile PTX at
    # runtime, so cuobjdump's resource usage would not describe what runs: warn and skip such a
    # device, keeping any that can be estimated, and only error out if none can.
    if args.compute_capability is not None or args.max_resident_threads is not None:
        if args.compute_capability is None or args.max_resident_threads is None:
            parser.error("--compute-capability and --max-resident-threads must be given together")
        devices = [{
            "index": None,
            "compute_capability": args.compute_capability,
            "name": "target device",
            "max_resident_threads": args.max_resident_threads,
        }]
    else:
        devices = detect_devices()
        if not devices:
            parser.error("no CUDA devices detected; pass --compute-capability and --max-resident-threads")
    if records:
        available_sms = _available_arches(records)
        satisfiable = []
        for device in devices:
            target_sm = compute_capability_to_sm(device["compute_capability"])
            if select_arch(target_sm, available_sms)[0] is not None:
                satisfiable.append(device)
                continue
            prefix = "device {}: ".format(device["index"]) if device.get("index") is not None else ""
            sys.stderr.write(
                "WARNING: {}{} (compute capability {}, {}) has no compatible embedded CUDA arch and "
                "would JIT-compile PTX at runtime."
                "\nSkipping it (its stack/shared-memory usage cannot be reported)."
                "\nEmbedded arches: {}.\n".format(
                    prefix, device.get("name", ""), device["compute_capability"], target_sm,
                    ", ".join(available_sms)))
        if not satisfiable:
            parser.error(
                "no target CUDA device can run the embedded arches ({}):"
                "\nevery device would JIT-compile PTX at runtime, so the result would be meaningless."
                "\nRebuild the CUDA libraries for a compatible architecture.".format(", ".join(available_sms)))
        devices = satisfiable

    exit_code = 0
    launched_names = None
    launched_modules = None
    launched_shared = None
    unmatched_display = None
    if args.launched:
        sys.stderr.write("Running cmsRun with the CUPTI kernel logger service to capture launched kernels...\n")
        launched_names, launched_modules, launched_shared, loaded_libraries, returncode = capture_launched_kernels(
            config, config_args)
        if returncode != 0:
            # the launched data is incomplete; report it and fail so automation notices
            sys.stderr.write("WARNING: cmsRun exited with code {}; launched-kernel data may be incomplete.\n".format(
                returncode))
            exit_code = 1
        if not launched_names:
            sys.stderr.write("WARNING: no launched kernels captured. If this was a GPU run the service may not "
                             "have attached (another CUPTI client active, a non-CUDA backend, or an abnormal "
                             "exit); the launched figures below will read 0 B.\n")
        # scan libraries the run loaded but the configuration did not, so launched kernels in
        # services or shared device libraries are found; has_device_code skips host-only ones.
        # realpath dedup avoids rescanning a configuration library reached via a different path.
        inspected = {os.path.realpath(path) for path in library_labels}
        extra = []
        for library in sorted(loaded_libraries):
            real = os.path.realpath(library)
            if real in inspected or not os.path.exists(library) or not has_device_code(library):
                continue
            inspected.add(real)
            extra.append(library)
        records += _scan_libraries(extra, cuobjdump, False)

    # only the spilling kernels (STACK > 0) are ever displayed (worst-stack headline and the
    # verbose listing), so demangle just those; the rest keep their mangled name as "short"
    names = demangle({record["kernel"] for record in records if record["stack"] > 0})
    for record in records:
        record["demangled"] = names.get(record["kernel"], record["kernel"])
        record["short"] = short_kernel_name(record["demangled"])
        # attach the launch-time dynamic shared memory (0 in static mode) and the per-block total
        dynamic_shared = launched_shared.get(record["kernel"], 0) if launched_shared else 0
        record["dyn_shared"] = dynamic_shared
        record["total_shared"] = record.get("shared", 0) + dynamic_shared

    if launched_names:
        unmatched = sorted(launched_names - {record["kernel"] for record in records})
        if unmatched:
            unmatched_names = demangle(unmatched)
            unmatched_display = sorted(short_kernel_name(unmatched_names.get(name, name)) for name in unmatched)

    _report(sys.stdout, records, library_labels, devices, unresolved, args.top, args.verbose,
            launched_names, unmatched_display, launched_modules)
    return exit_code
