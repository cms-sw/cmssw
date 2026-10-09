#!/usr/bin/env python3
# Original author: Felice Pantaleo, felice.pantaleo@cern.ch, 02/2026
"""Dump the static per-kernel resource usage (ptxas info) of CUDA/Alpaka libraries.

Runs `cuobjdump --dump-resource-usage` on each given shared object / cubin and
reports, per kernel, the registers/thread, stack frame, static shared memory,
local memory (>0 means register spills) and constant memory -- the same numbers
ptxas prints with `-Xptxas -v`. Names are demangled with `c++filt`/`cu++filt`.

This is the compile-time counterpart of the runtime CUPTI layer in
PerfettoTraceService (traceGpuKernels=True): CUPTI reports registers and memory
per launch, while this works offline on the built libraries and also exposes
spills/stack, which are not in the CUPTI kernel records.

Usage:
  perfettoKernelResources.py [--json] [--filter SUBSTR] <lib.so|cubin> [more ...]
"""
import argparse
import json
import re
import shutil
import subprocess
import sys

_FUNC = re.compile(r"^\s*Function (.+):\s*$")
_RES = re.compile(r"REG:(\d+).*?STACK:(\d+).*?SHARED:(\d+).*?LOCAL:(\d+)")
_CONST = re.compile(r"CONSTANT\[0\]:(\d+)")


def demangle(name, tool):
    if not tool:
        return name
    out = subprocess.run([tool, name], capture_output=True, text=True).stdout.strip()
    return out or name


def shorten(name):
    """Reduce an Alpaka gpuKernel<...> wrapper to the user functor."""
    i = name.find("gpuKernel<")
    if i < 0:
        return name
    i += len("gpuKernel<")
    depth = 0
    for j in range(i, len(name)):
        c = name[j]
        if c == "<":
            depth += 1
        elif c == ">":
            if depth == 0:
                return name[i:j]
            depth -= 1
        elif c == "," and depth == 0:
            return name[i:j]
    return name[i:]


def dump(path, demangler):
    out = subprocess.run(["cuobjdump", "--dump-resource-usage", path], capture_output=True, text=True).stdout
    kernels, cur = [], None
    for line in out.splitlines():
        m = _FUNC.match(line)
        if m:
            cur = m.group(1)
            continue
        if cur:
            r = _RES.search(line)
            if r:
                c = _CONST.search(line)
                full = demangle(cur, demangler)
                kernels.append({
                    "kernel": shorten(full),
                    "registers": int(r.group(1)),
                    "stack": int(r.group(2)),
                    "shared": int(r.group(3)),
                    "local_spill": int(r.group(4)),
                    "const": int(c.group(1)) if c else 0,
                    "mangled": cur,
                })
                cur = None
    return kernels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("libs", nargs="+", help="shared objects or cubins to inspect")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    ap.add_argument("--filter", default="", help="only keep kernels whose name contains SUBSTR")
    args = ap.parse_args()

    if not shutil.which("cuobjdump"):
        sys.exit("cuobjdump not found; run inside a CUDA environment (cmsenv)")
    demangler = shutil.which("cu++filt") or shutil.which("c++filt")

    kernels = []
    for lib in args.libs:
        for k in dump(lib, demangler):
            if args.filter in k["kernel"]:
                kernels.append(k)
    kernels.sort(key=lambda k: -k["registers"])

    if args.json:
        print(json.dumps(kernels, indent=2))
        return
    print(f"{'REG':>4} {'STACK':>6} {'SHARED':>7} {'SPILL':>6} {'CONST':>6}  kernel")
    for k in kernels:
        print(f"{k['registers']:>4} {k['stack']:>6} {k['shared']:>7} {k['local_spill']:>6} {k['const']:>6}  "
              f"{k['kernel'][:90]}")


if __name__ == "__main__":
    main()
