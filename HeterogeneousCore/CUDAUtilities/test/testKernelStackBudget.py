#!/usr/bin/env python3
"""Unit tests for kernelStackBudget.

These cover the pure parsing/helper functions and, with the heavy I/O stubbed (cuobjdump,
cmsRun, the configuration and the demangler), the end-to-end main() path: CLI splitting,
option validation, compute-capability normalisation, exit codes and the report.
"""
import contextlib
import io
import os
import tempfile
import types
import unittest
from unittest import mock

from HeterogeneousCore.CUDAUtilities import kernelStackBudget as ksb


# a realistic cuobjdump --dump-resource-usage excerpt (two arches, two kernels)
RESOURCE_USAGE = """
Fatbin elf code:
================
arch = sm_80
identifier = foo.cu.o

Resource usage:
 Common:
  GLOBAL:216
 Function _Z6kernelv:
  REG:26 STACK:80 SHARED:0 LOCAL:0 CONSTANT[0]:352 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _Z9spillsLotv:
  REG:57 STACK:112 SHARED:10144 LOCAL:48 CONSTANT[0]:0 TEXTURE:0 SURFACE:0 SAMPLER:0

Fatbin elf code:
================
arch = sm_90
identifier = foo.cu.o

Resource usage:
 Common:
  GLOBAL:216
 Function _Z6kernelv:
  REG:24 STACK:16 SHARED:0 LOCAL:0 CONSTANT[0]:352 TEXTURE:0 SURFACE:0 SAMPLER:0
"""

# a cudaComputeCapabilities --verbose excerpt
DEVICE_VERBOSE = """\
   0     9.0    NVIDIA H100 NVL
        multiprocessors:                                          132
        max threads per multiprocessor:                          2048
        max resident threads:                                  270336
   1     8.0    NVIDIA A100 (unsupported)
        max resident threads:                                  221184
"""


class ResourceUsageTest(unittest.TestCase):
    def test_parses_per_arch_kernels(self):
        records = ksb.parse_resource_usage(RESOURCE_USAGE)
        self.assertEqual(len(records), 3)
        sm80 = [r for r in records if r["arch"] == "sm_80"]
        self.assertEqual(max(r["stack"] for r in sm80), 112)
        self.assertEqual(max(r["local"] for r in sm80), 48)
        sm90 = [r for r in records if r["arch"] == "sm_90"]
        self.assertEqual(max(r["stack"] for r in sm90), 16)

    def test_empty_input(self):
        self.assertEqual(ksb.parse_resource_usage(""), [])

    def test_parses_arch_variant_suffix(self):
        # cuobjdump reports the CUDA 12.9+ 'a'/'f' variant suffix, which must be preserved
        text = ("arch = sm_90a\n"
                " Function _Zk:\n"
                "  REG:10 STACK:8 SHARED:0 LOCAL:0 CONSTANT[0]:0 TEXTURE:0 SURFACE:0 SAMPLER:0\n")
        records = ksb.parse_resource_usage(text)
        self.assertEqual(records[0]["arch"], "sm_90a")


class DeviceInfoTest(unittest.TestCase):
    def test_parses_devices_and_max_resident(self):
        devices = ksb.parse_device_info(DEVICE_VERBOSE)
        self.assertEqual(len(devices), 2)
        self.assertEqual(devices[0]["compute_capability"], "9.0")
        self.assertEqual(devices[0]["max_resident_threads"], 270336)
        self.assertEqual(devices[0]["name"], "NVIDIA H100 NVL")
        # "(unsupported)" must not bleed into the device name
        self.assertEqual(devices[1]["name"], "NVIDIA A100")
        self.assertEqual(devices[1]["max_resident_threads"], 221184)

    def test_detail_lines_are_not_parsed_as_devices(self):
        # only the two summary lines start a device; the indented detail rows (which carry
        # numbers too) must not be mistaken for additional devices
        devices = ksb.parse_device_info(DEVICE_VERBOSE)
        self.assertEqual([d["index"] for d in devices], [0, 1])
        self.assertEqual([d["compute_capability"] for d in devices], ["9.0", "8.0"])


class ArchTest(unittest.TestCase):
    def test_compute_capability_to_sm(self):
        self.assertEqual(ksb.compute_capability_to_sm("9.0"), "sm_90")
        self.assertEqual(ksb.compute_capability_to_sm("90"), "sm_90")
        self.assertEqual(ksb.compute_capability_to_sm("12.0"), "sm_120")
        self.assertEqual(ksb.compute_capability_to_sm("sm_90"), "sm_90")

    def test_dotted_sm_capability_normalises(self):
        # regression: 'sm_9.0' must become 'sm_90', not pass through and later break int()
        self.assertEqual(ksb.compute_capability_to_sm("sm_9.0"), "sm_90")
        target, fell_back = ksb.select_arch(ksb.compute_capability_to_sm("sm_9.0"), ["sm_80", "sm_90"])
        self.assertEqual((target, fell_back), ("sm_90", False))

    def test_select_arch(self):
        available = ["sm_70", "sm_80", "sm_90"]
        self.assertEqual(ksb.select_arch("sm_90", available), ("sm_90", False))
        self.assertEqual(ksb.select_arch("sm_89", available), ("sm_80", True))
        self.assertEqual(ksb.select_arch("sm_50", available), (None, False))

    def test_select_arch_respects_major_generation(self):
        # SASS is compatible only within a major generation and forward across minor revisions:
        # Turing (sm_75) runs Volta (sm_70) code, but not Pascal (sm_6x, older major)
        self.assertEqual(ksb.select_arch("sm_75", ["sm_70"]), ("sm_70", True))
        self.assertEqual(ksb.select_arch("sm_75", ["sm_60", "sm_61"]), (None, False))
        # a newer major is never selected for an older device
        self.assertEqual(ksb.select_arch("sm_80", ["sm_90"]), (None, False))
        # within a major, the highest minor <= the device's is chosen (sm_86 runs on sm_89)
        self.assertEqual(ksb.select_arch("sm_89", ["sm_80", "sm_86"]), ("sm_86", True))
        # a higher minor of the same major is not backward compatible (sm_86 does not run on sm_80)
        self.assertEqual(ksb.select_arch("sm_80", ["sm_86"]), (None, False))
        # two-digit majors split correctly (sm_100 -> major 10, minor 0)
        self.assertEqual(ksb._sm_major_minor("sm_100"), (10, 0))

    def test_parse_arch_variants(self):
        self.assertEqual(ksb._parse_arch("sm_90"), (90, ""))
        self.assertEqual(ksb._parse_arch("sm_90a"), (90, "a"))
        self.assertEqual(ksb._parse_arch("sm_100f"), (100, "f"))
        self.assertEqual(ksb._parse_arch("sm_120"), (120, ""))

    def test_select_arch_family_and_accelerated_variants(self):
        # base sm_100 runs across the Blackwell generation, including consumer sm_120
        self.assertEqual(ksb.select_arch("sm_103", ["sm_100"]), ("sm_100", True))
        self.assertEqual(ksb.select_arch("sm_120", ["sm_100"]), ("sm_100", True))
        # family-specific sm_100f stays within its major generation (sm_10x), not sm_120
        self.assertEqual(ksb.select_arch("sm_103", ["sm_100f"]), ("sm_100f", True))
        self.assertEqual(ksb.select_arch("sm_120", ["sm_100f"]), (None, False))
        # accelerated sm_100a runs only on exactly sm_100 (exact match is not a fallback)
        self.assertEqual(ksb.select_arch("sm_100", ["sm_100a"]), ("sm_100a", False))
        self.assertEqual(ksb.select_arch("sm_103", ["sm_100a"]), (None, False))
        # sm_90a (Hopper accelerated) runs on sm_90 but not sm_100
        self.assertEqual(ksb.select_arch("sm_90", ["sm_90a"]), ("sm_90a", False))
        self.assertEqual(ksb.select_arch("sm_100", ["sm_90a"]), (None, False))
        # among compatible arches the highest compute capability is chosen
        self.assertEqual(ksb.select_arch("sm_120", ["sm_100", "sm_103"]), ("sm_103", True))


class DeviceRecordsTest(unittest.TestCase):
    def _rec(self, arch, kernel="_Zk", library="libA.so", stack=0):
        return {"arch": arch, "kernel": kernel, "library": library, "stack": stack}

    def test_prefers_accelerated_then_family_then_base(self):
        # all three variants of the device's own CC are compatible; the accelerated one wins
        picked = ksb._device_records([self._rec("sm_100"), self._rec("sm_100f"), self._rec("sm_100a")], 100)
        self.assertEqual([r["arch"] for r in picked], ["sm_100a"])

    def test_family_preferred_over_base_even_with_lower_cc(self):
        # device sm_103: sm_100a is incompatible (needs exactly sm_100); family sm_100f beats base
        # sm_103 despite the lower compute capability, per the a > f > base preference
        picked = ksb._device_records([self._rec("sm_100f"), self._rec("sm_103"), self._rec("sm_100a")], 103)
        self.assertEqual([r["arch"] for r in picked], ["sm_100f"])

    def test_highest_cc_wins_within_the_same_variant(self):
        # device sm_89: sm_80 and sm_86 are both compatible base arches; the higher one is chosen
        picked = ksb._device_records([self._rec("sm_80"), self._rec("sm_86")], 89)
        self.assertEqual([r["arch"] for r in picked], ["sm_86"])

    def test_drops_functions_with_no_compatible_arch(self):
        # sm_80 does not run on an sm_90 device (different major); the function would JIT, so drop it
        picked = ksb._device_records([self._rec("sm_80", kernel="_Zonly80")], 90)
        self.assertEqual(picked, [])


class ShortNameTest(unittest.TestCase):
    def test_extracts_inner_alpaka_kernel(self):
        name = ("void alpaka::detail::gpuKernel<alpaka_cuda_async::TestAlgoKernelUpdate, "
                "alpaka::AccGpuUniformCudaHipRt<alpaka::ApiCudaRt, std::integral_constant<unsigned long, 1ul> > >"
                "(alpaka::Vec<...>, ...)")
        self.assertEqual(ksb.short_kernel_name(name), "alpaka_cuda_async::TestAlgoKernelUpdate")

    def test_passthrough_for_plain_kernel(self):
        self.assertEqual(ksb.short_kernel_name("myPlainKernel(int*)"), "myPlainKernel(int*)")


class _FakeBackend:
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value


class _FakeAlpaka:
    def __init__(self, backend):
        self.backend = _FakeBackend(backend)


class _FakeModule:
    def __init__(self, backend=None):
        if backend is not None:
            self.alpaka = _FakeAlpaka(backend)


class PluginNameTest(unittest.TestCase):
    def test_plain_type_is_its_own_plugin(self):
        self.assertEqual(ksb.plugin_name_for("FooProducer", _FakeModule(), "cuda_async"), "FooProducer")

    def test_alpaka_default_backend(self):
        self.assertEqual(
            ksb.plugin_name_for("Foo@alpaka", _FakeModule(), "cuda_async"),
            "alpaka_cuda_async::Foo",
        )

    def test_alpaka_explicit_backend_wins(self):
        self.assertEqual(
            ksb.plugin_name_for("Foo@alpaka", _FakeModule("serial_sync"), "cuda_async"),
            "alpaka_serial_sync::Foo",
        )


class ProcessAcceleratorAlpaka:
    """Fake ProcessAcceleratorAlpaka; config_backend matches it by type(...).__name__."""

    def __init__(self, backend=None):
        self._backend = backend


class _FakeProcess:
    def __init__(self, accelerators=(), option_accelerators=("*",)):
        self._accelerators = {type(a).__name__: a for a in accelerators}
        self.options = types.SimpleNamespace(accelerators=list(option_accelerators))

    def processAccelerators_(self):
        return self._accelerators


class ConfigBackendTest(unittest.TestCase):
    def test_defaults_to_cuda_async(self):
        self.assertEqual(ksb.config_backend(_FakeProcess()), "cuda_async")

    def test_explicit_set_backend_wins(self):
        proc = _FakeProcess(accelerators=[ProcessAcceleratorAlpaka("serial_sync")])
        self.assertEqual(ksb.config_backend(proc), "serial_sync")

    def test_accelerators_cpu_selects_serial_sync(self):
        self.assertEqual(ksb.config_backend(_FakeProcess(option_accelerators=("cpu",))), "serial_sync")

    def test_accelerators_gpu_pattern_selects_cuda(self):
        self.assertEqual(ksb.config_backend(_FakeProcess(option_accelerators=("gpu-*",))), "cuda_async")

    def test_unreadable_process_falls_back_to_cuda_async(self):
        self.assertEqual(ksb.config_backend(object()), "cuda_async")


class LibDirectoriesTest(unittest.TestCase):
    def test_derives_from_ld_library_path_and_covers_microarch(self):
        with tempfile.TemporaryDirectory() as root:
            arch = "el9_amd64_gcc14"
            dev = os.path.join(root, "dev", "lib", arch)
            microarch = os.path.join(dev, "x86-64-v3")          # micro-architecture subdirectory
            base = os.path.join(root, "base", "lib", arch)
            external = os.path.join(root, "dev", "external", arch, "lib")  # must be skipped
            for directory in (microarch, base, external):
                os.makedirs(directory)
            ld = os.pathsep.join([microarch, dev, external, base, "/opt/cuda/lib64"])
            with mock.patch.dict(os.environ, {"SCRAM_ARCH": arch, "LD_LIBRARY_PATH": ld}, clear=True):
                dirs = ksb.lib_directories()
            # kept: lib/$ARCH and its micro-arch subdir, developer area first; dropped: externals
            self.assertEqual(dirs, [microarch, dev, base])


class PluginCacheTest(unittest.TestCase):
    def test_parses_name_to_library(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, ".edmplugincache"), "w") as handle:
                handle.write("pluginFooCudaAsync.so alpaka_cuda_async::Foo CMS%EDM%Framework%Module\n")
                handle.write("pluginFooCudaAsync.so alpaka_cuda_async::Foo CMS%EDM%Framework%ParameterSet%Description\n")
                handle.write("pluginBar.so BarProducer CMS%EDM%Framework%Module\n")
            mapping = ksb.parse_plugin_cache([directory])
        self.assertEqual(mapping["alpaka_cuda_async::Foo"], "pluginFooCudaAsync.so")
        self.assertEqual(mapping["BarProducer"], "pluginBar.so")


class ReadLaunchedTest(unittest.TestCase):
    def test_parses_kernels_module_and_dynamic_shared(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "kernels.txt")
            with open(path, "w") as handle:
                handle.write("_Zk1\tmodA\t2048\n")
                handle.write("_Zk1\tmodB\t4096\n")  # second module, larger dynamic shared memory
                handle.write("_Zk2\t\t0\n")          # no module, no dynamic shared
                handle.write("\n")                    # blank line ignored
            names, modules, dynamic_shared = ksb._read_launched(path)
        self.assertEqual(names, {"_Zk1", "_Zk2"})
        self.assertEqual(modules, {"_Zk1": ["modA", "modB"]})  # sorted, deduped; _Zk2 has none
        self.assertEqual(dynamic_shared, {"_Zk1": 4096})       # max across launches; _Zk2 zero -> absent

    def test_missing_file_is_empty(self):
        names, modules, dynamic_shared = ksb._read_launched("/no/such/file")
        self.assertEqual(names, set())
        self.assertEqual(modules, {})
        self.assertEqual(dynamic_shared, {})

    def test_reads_legacy_two_column_lines(self):
        # logs written by an older service (before the dynamic-shared column) still parse
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "kernels.txt")
            with open(path, "w") as handle:
                handle.write("_Zk1\tmodA\n")
            names, modules, dynamic_shared = ksb._read_launched(path)
        self.assertEqual(names, {"_Zk1"})
        self.assertEqual(modules, {"_Zk1": ["modA"]})
        self.assertEqual(dynamic_shared, {})

    def test_read_lines_strips_and_dedups(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "libs.txt")
            with open(path, "w") as handle:
                handle.write("/a/libX.so\n\n  /a/libY.so  \n/a/libX.so\n")
            self.assertEqual(ksb._read_lines(path), {"/a/libX.so", "/a/libY.so"})


class ValueOptionsTest(unittest.TestCase):
    def test_value_options_derived_from_parser(self):
        options = ksb._value_option_strings(ksb._build_parser())
        self.assertEqual(options, {"--compute-capability", "--max-resident-threads", "--top"})

    def test_store_true_flags_are_not_value_options(self):
        options = ksb._value_option_strings(ksb._build_parser())
        self.assertNotIn("--launched", options)
        self.assertNotIn("--verbose", options)

    def test_value_option_consumes_following_token(self):
        options = ksb._value_option_strings(ksb._build_parser())
        tool, config, forwarded = ksb._split_cli(["--compute-capability", "9.0", "cfg.py"], options)
        self.assertEqual(tool, ["--compute-capability", "9.0"])
        self.assertEqual(config, "cfg.py")
        self.assertEqual(forwarded, [])

    def test_abbreviated_flag_is_rejected(self):
        # regression: '--comp' used to be swallowed by _split_cli while the real config was
        # forwarded away; with allow_abbrev off the parser now rejects it cleanly
        parser = ksb._build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--comp", "9.0", "cfg.py"])


class ReportTest(unittest.TestCase):
    def _records(self):
        # the kernel with the biggest stack frame (_ZbigA, 1024 B) is NOT launched;
        # only the smaller _Zsmall (64 B) is, so the launched budget must be lower
        return [
            {"arch": "sm_90", "kernel": "_ZbigA", "stack": 1024, "local": 0, "shared": 8192, "reg": 40,
             "library": "libX.so", "demangled": "BigA", "short": "BigA"},
            {"arch": "sm_90", "kernel": "_Zsmall", "stack": 64, "local": 0, "shared": 1024, "reg": 20,
             "library": "libX.so", "demangled": "Small", "short": "Small"},
        ]

    def _devices(self):
        return [{"index": None, "compute_capability": "9.0", "name": "dev", "max_resident_threads": 1000}]

    def test_static_mode_uses_largest_kernel(self):
        out = io.StringIO()
        ksb._report(out, self._records(), {"libX.so": ["m"]}, self._devices(), [], 25, False)
        text = out.getvalue()
        self.assertIn("worst-case reservation: ", text)
        self.assertIn("1024000 B", text)  # 1000 threads x 1024 B

    def test_launched_mode_lowers_budget_to_launched_kernels(self):
        out = io.StringIO()
        ksb._report(out, self._records(), {"libX.so": ["m"]}, self._devices(), [], 25, False,
                    launched_names={"_Zsmall"})
        text = out.getvalue()
        self.assertIn("kernels actually launched (CUPTI):      1 (1 matched to stack data)", text)
        self.assertIn("reservation (launched kernels): ", text)
        self.assertIn("64000 B", text)    # 1000 threads x 64 B (launched)
        self.assertIn("static upper bound", text)
        self.assertIn("1024000 B", text)  # 1000 threads x 1024 B (all kernels)

    def test_launched_mode_with_no_kernels_is_zero(self):
        out = io.StringIO()
        ksb._report(out, self._records(), {"libX.so": ["m"]}, self._devices(), [], 25, False,
                    launched_names=set())
        text = out.getvalue()
        self.assertIn("kernels actually launched (CUPTI):      0 (0 matched to stack data)", text)
        self.assertIn("reservation (launched kernels): ", text)
        self.assertIn("(0 B)", text)       # the launched reservation is zero
        self.assertIn("static upper bound", text)
        self.assertIn("1024000 B", text)   # but the static bound still covers every kernel

    def test_launched_mode_reports_module_attribution(self):
        out = io.StringIO()
        ksb._report(out, self._records(), {"libX.so": ["m"]}, self._devices(), [], 25, False,
                    launched_names={"_Zsmall"}, launched_modules={"_Zsmall": ["modA", "modB"]})
        self.assertIn("[modA, modB]", out.getvalue())

    def test_static_mode_reports_worst_stack_kernel_static_shared(self):
        out = io.StringIO()
        ksb._report(out, self._records(), {"libX.so": ["m"]}, self._devices(), [], 25, False)
        text = out.getvalue()
        # the worst-case reservation is _ZbigA (largest stack, 1024 B); its static shared is reported
        self.assertIn("worst-case reservation", text)
        self.assertIn("shared memory: 8192 B static", text)

    def test_launched_mode_reports_worst_stack_kernel_shared(self):
        records = self._records()
        # attach dynamic shared as main() would; the worst-STACK launched kernel (_ZbigA) requests
        # 16384 B of dynamic shared memory, reported together with its 8192 B static amount. Note the
        # figures describe _ZbigA (largest stack), not _Zsmall which happens to use less shared memory
        for record in records:
            record["dyn_shared"] = 16384 if record["kernel"] == "_ZbigA" else 0
            record["total_shared"] = record["shared"] + record["dyn_shared"]
        out = io.StringIO()
        ksb._report(out, records, {"libX.so": ["m"]}, self._devices(), [], 25, False,
                    launched_names={"_ZbigA", "_Zsmall"})
        text = out.getvalue()
        self.assertIn("reservation (launched kernels)", text)
        self.assertIn("shared memory: 24576 B total = 8192 B static + 16384 B dynamic", text)

    def test_arch_variant_kernels_counted_on_compatible_device(self):
        # a base sm_90 kernel and an accelerated sm_90a kernel both run on an sm_90 device; the
        # sm_90a one must still be counted even though the device's plain arch is sm_90
        records = [
            {"arch": "sm_90", "kernel": "_Zbase", "stack": 100, "local": 0, "shared": 0, "reg": 20,
             "library": "libA.so", "demangled": "Base", "short": "Base"},
            {"arch": "sm_90a", "kernel": "_Zaccel", "stack": 900, "local": 0, "shared": 0, "reg": 40,
             "library": "libB.so", "demangled": "Accel", "short": "Accel"},
        ]
        out = io.StringIO()
        ksb._report(out, records, {"libA.so": ["m"], "libB.so": ["n"]}, self._devices(), [], 25, False)
        text = out.getvalue()
        # the sm_90a kernel (900 B stack) drives the reservation on the sm_90 device
        self.assertIn("largest stack frame 900 B", text)
        self.assertIn("kernel: Accel", text)

    def test_verbose_shows_dynamic_shared_column_only_when_launched(self):
        records = self._records()
        for record in records:
            record["dyn_shared"] = 512 if record["kernel"] == "_Zsmall" else 0
            record["total_shared"] = record["shared"] + record["dyn_shared"]
        launched = io.StringIO()
        ksb._report(launched, records, {"libX.so": ["m"]}, self._devices(), [], 25, True,
                    launched_names={"_ZbigA", "_Zsmall"})
        self.assertIn("SHARED", launched.getvalue())
        self.assertIn("DYN", launched.getvalue())
        static = io.StringIO()
        ksb._report(static, self._records(), {"libX.so": ["m"]}, self._devices(), [], 25, True)
        self.assertIn("SHARED", static.getvalue())     # static shared column always present
        self.assertNotIn("DYN", static.getvalue())     # dynamic column only in launched mode


class SplitCliTest(unittest.TestCase):
    def test_options_before_config_and_forwarded_args(self):
        tool, config, forwarded = ksb._split_cli(["--top", "5", "-v", "cfg.py", "--run", "2"])
        self.assertEqual(tool, ["--top", "5", "-v"])
        self.assertEqual(config, "cfg.py")
        self.assertEqual(forwarded, ["--run", "2"])

    def test_equals_form_value_option(self):
        tool, config, forwarded = ksb._split_cli(["--compute-capability=9.0", "cfg.py"])
        self.assertEqual(tool, ["--compute-capability=9.0"])
        self.assertEqual(config, "cfg.py")
        self.assertEqual(forwarded, [])

    def test_no_config(self):
        tool, config, forwarded = ksb._split_cli(["--help"])
        self.assertEqual(config, None)


class MainIntegrationTest(unittest.TestCase):
    """Drive main() end-to-end with cuobjdump, cmsRun, the config and the demangler stubbed,
    so CLI splitting, validation, arch normalisation, exit codes and the report run for real."""

    _RECORDS = [
        {"arch": "sm_90", "kernel": "_Zbig", "stack": 512, "local": 0, "shared": 2048, "reg": 40},
        {"arch": "sm_90", "kernel": "_Zsmall", "stack": 64, "local": 0, "shared": 256, "reg": 20},
    ]

    def _run(self, argv, **extra):
        def scan(paths, cuobjdump, from_config):
            if not paths:  # mirror _scan_libraries: nothing to scan -> no records
                return []
            return [dict(record, library="libX.so", config=from_config) for record in self._RECORDS]

        stubs = dict(
            processFromFile=lambda config, args: object(),
            find_tool=lambda name: "/bin/true",
            loaded_cuda_libraries=lambda *a, **k: ({"libX.so": ["m"]}, []),
            _scan_libraries=scan,
            demangle=lambda names: {name: name for name in names},
        )
        stubs.update(extra)
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.multiple(ksb, **stubs), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = ksb.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_static_report_accepts_dotted_sm_capability(self):
        # regression: '--compute-capability sm_9.0' used to crash with an uncaught ValueError
        code, text, _ = self._run(
            ["--compute-capability", "sm_9.0", "--max-resident-threads", "1000", "cfg.py"])
        self.assertEqual(code, 0)
        self.assertIn("worst-case reservation", text)
        self.assertIn("512000 B", text)  # 1000 threads x 512 B (largest stack frame)

    def test_launched_failure_propagates_nonzero_exit(self):
        # regression: a failed cmsRun used to still exit 0 and hide the failure from automation
        code, _, err = self._run(
            ["--launched", "--compute-capability", "90", "--max-resident-threads", "1000", "cfg.py"],
            capture_launched_kernels=lambda config, args: (set(), {}, {}, set(), 42))
        self.assertNotEqual(code, 0)
        self.assertIn("cmsRun exited with code 42", err)

    def test_invalid_compute_capability_is_rejected(self):
        # a non-numeric capability is caught by the compute_capability_to_sm normalisation
        with self.assertRaises(SystemExit):
            self._run(["--compute-capability", "abc", "--max-resident-threads", "1000", "cfg.py"])

    def test_incompatible_device_arch_exits_with_error(self):
        # the libraries embed sm_90 SASS (per _RECORDS) but the only target is sm_50 (Maxwell): no
        # compatible arch and no other device, so the tool must exit with an error
        with self.assertRaises(SystemExit):
            self._run(["--compute-capability", "50", "--max-resident-threads", "1000", "cfg.py"])

    def test_partially_compatible_devices_warn_and_continue(self):
        # two detected GPUs: sm_90 (matches the embedded sm_90) and sm_50 (no compatible arch). The
        # incompatible one is warned about and skipped; the tool still reports the compatible one.
        dev90 = {"index": 0, "compute_capability": "9.0", "name": "GPU90", "max_resident_threads": 1000}
        dev50 = {"index": 1, "compute_capability": "5.0", "name": "GPU50", "max_resident_threads": 500}
        code, text, err = self._run(["cfg.py"], detect_devices=lambda: [dev90, dev50])
        self.assertEqual(code, 0)
        self.assertIn("sm_50", err)         # warned about the incompatible device
        self.assertIn("GPU90", text)        # still reported the compatible one
        self.assertNotIn("GPU50", text)     # the incompatible one is skipped in the report

    def test_launched_reports_dynamic_shared_end_to_end(self):
        # a successful launched run where _Zbig requests 4096 B of dynamic shared memory; main()
        # must attach it to the record and the report must combine it with the 2048 B static amount
        code, text, _ = self._run(
            ["--launched", "--compute-capability", "90", "--max-resident-threads", "1000", "cfg.py"],
            capture_launched_kernels=lambda config, args: ({"_Zbig"}, {"_Zbig": ["m"]}, {"_Zbig": 4096}, set(), 0))
        self.assertEqual(code, 0)
        # _Zbig is the largest-stack launched kernel; its shared memory is reported in the block
        self.assertIn("shared memory: 6144 B total = 2048 B static + 4096 B dynamic", text)

    def test_non_cuda_backend_exits_early(self):
        # a CPU/serial (or AMD) configuration launches no CUDA kernels: fail fast with an error
        # instead of running to the end and reporting an empty budget
        with self.assertRaises(SystemExit):
            self._run(["--compute-capability", "90", "--max-resident-threads", "1000", "cfg.py"],
                      config_backend=lambda process: "serial_sync")


if __name__ == "__main__":
    # verbose so every check is listed by name: the abbreviated-flag test deliberately makes
    # argparse print a usage/error message, and showing each result makes clear it belongs to
    # that (passing) test rather than signalling a failure
    unittest.main(verbosity=2)
