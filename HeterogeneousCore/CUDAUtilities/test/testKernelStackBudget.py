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
    def test_parses_kernels_and_module_attribution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "kernels.txt")
            with open(path, "w") as handle:
                handle.write("_Zk1\tmodA\n")
                handle.write("_Zk1\tmodB\n")  # same kernel launched by a second module
                handle.write("_Zk2\t\n")       # launched with no attributed module
                handle.write("\n")             # blank line ignored
            names, modules = ksb._read_launched(path)
        self.assertEqual(names, {"_Zk1", "_Zk2"})
        self.assertEqual(modules, {"_Zk1": ["modA", "modB"]})  # sorted, deduped; _Zk2 has none

    def test_missing_file_is_empty(self):
        names, modules = ksb._read_launched("/no/such/file")
        self.assertEqual(names, set())
        self.assertEqual(modules, {})

    def test_read_lines_strips_and_dedups(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "libs.txt")
            with open(path, "w") as handle:
                handle.write("/a/libX.so\n\n  /a/libY.so  \n/a/libX.so\n")
            self.assertEqual(ksb._read_lines(path), {"/a/libX.so", "/a/libY.so"})


class ValueOptionsTest(unittest.TestCase):
    def test_value_options_derived_from_parser(self):
        options = ksb._value_option_strings(ksb._build_parser())
        self.assertEqual(options, {"--backend", "--compute-capability", "--max-resident-threads", "--top"})

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
            {"arch": "sm_90", "kernel": "_ZbigA", "stack": 1024, "local": 0, "reg": 40,
             "library": "libX.so", "demangled": "BigA", "short": "BigA"},
            {"arch": "sm_90", "kernel": "_Zsmall", "stack": 64, "local": 0, "reg": 20,
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


class SplitCliTest(unittest.TestCase):
    def test_options_before_config_and_forwarded_args(self):
        tool, config, forwarded = ksb._split_cli(["--top", "5", "-v", "cfg.py", "--run", "2"])
        self.assertEqual(tool, ["--top", "5", "-v"])
        self.assertEqual(config, "cfg.py")
        self.assertEqual(forwarded, ["--run", "2"])

    def test_equals_form_value_option(self):
        tool, config, forwarded = ksb._split_cli(["--backend=serial_sync", "cfg.py"])
        self.assertEqual(tool, ["--backend=serial_sync"])
        self.assertEqual(config, "cfg.py")
        self.assertEqual(forwarded, [])

    def test_no_config(self):
        tool, config, forwarded = ksb._split_cli(["--help"])
        self.assertEqual(config, None)


class MainIntegrationTest(unittest.TestCase):
    """Drive main() end-to-end with cuobjdump, cmsRun, the config and the demangler stubbed,
    so CLI splitting, validation, arch normalisation, exit codes and the report run for real."""

    _RECORDS = [
        {"arch": "sm_90", "kernel": "_Zbig", "stack": 512, "local": 0, "reg": 40},
        {"arch": "sm_90", "kernel": "_Zsmall", "stack": 64, "local": 0, "reg": 20},
    ]

    def _run(self, argv, **extra):
        def scan(paths, cuobjdump, from_config):
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
            capture_launched_kernels=lambda config, args: (set(), {}, set(), 42))
        self.assertNotEqual(code, 0)
        self.assertIn("cmsRun exited with code 42", err)


if __name__ == "__main__":
    # verbose so every check is listed by name: the abbreviated-flag test deliberately makes
    # argparse print a usage/error message, and showing each result makes clear it belongs to
    # that (passing) test rather than signalling a failure
    unittest.main(verbosity=2)
