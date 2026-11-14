#!/usr/bin/env python3
"""
Produce a single, fully self-contained EDM config dump of a Phase-2 L1T
Global Trigger "menu" cff (e.g. l1tGTMenu_cff.py), suitable for browsing in
ConfDB: every cff import is resolved and every PSet/Path/producer is
inlined, exactly like `process.dumpPython()` does for a full process.

Normal usage (inside `cmsenv`, where FWCore.ParameterSet.Config is already
importable):

    python3 dumpL1TGTMenu.py

Outside a CMSSW runtime (e.g. a bare git checkout with no scram/cvmfs, such
as this repo in a sandbox), FWCore.ParameterSet.Config and friends are pure
Python and don't need the compiled framework to just build/dump a Process,
so this script builds a minimal on-demand "python farm" -- a symlink tree
mirroring only the CMSSW packages actually imported by the menu chain -- by
retrying the dump in a subprocess, mirroring whichever package a
ModuleNotFoundError names, until it succeeds. Nothing outside a scratch
temp dir is touched.

    --src           path to a CMSSW source checkout (defaults to
                     $CMSSW_BASE/src, then the git toplevel containing this
                     script)
    --menu          dotted module path of the menu cff to dump
                     (default: the step1_2024 GT menu)
    --gtemulator    dotted module path providing the GT producer +
                     algo-block producer (default: GTemulator_cff)
    --out           output .py file (default: derived from --menu)
    --flat          inline the object definitions and compute the online
                     thresholds (L1Trigger.Phase2L1GT.l1tGTMenuTools.flattenGTMenu),
                     i.e. the configuration as seen by the emulator. By default
                     the structured form is kept: conditions reference the shared
                     object/scaling PSets via refToPSet_, which is what should be
                     loaded into ConfDB.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

DEFAULT_MENU = "L1Trigger.Configuration.Phase2GTMenus.SeedDefinitions.step1_2024.l1tGTMenu_cff"
DEFAULT_GTEMULATOR = "L1Trigger.Configuration.GTemulator_cff"
MAX_MIRROR_ITERATIONS = 30

DRIVER_TEMPLATE = '''\
import os
import FWCore.ParameterSet.Config as cms

process = cms.Process("L1GTMenuDump")

process.load({gtemulator!r})
process.GTemulation_step = cms.Path(process.GTemulator)

process.load({menu!r})

from L1Trigger.Phase2L1GT.l1tGTAlgoBlockProducer_cff import collectAlgorithmPaths
process.schedule = cms.Schedule(process.GTemulation_step, *collectAlgorithmPaths(process))

if {flat!r}:
    from L1Trigger.Phase2L1GT.l1tGTMenuTools import flattenGTMenu
    flattenGTMenu(process)

with open(os.environ["DUMP_OUT_FILE"], "w") as f:
    f.write(process.dumpPython())

print("DUMP_OK paths=%d" % len(process.schedule))
'''


def find_default_src_root(script_path):
    cmssw_base = os.environ.get("CMSSW_BASE")
    if cmssw_base and os.path.isdir(os.path.join(cmssw_base, "src")):
        return os.path.join(cmssw_base, "src")
    try:
        top = subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(script_path)), "rev-parse", "--show-toplevel"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if top:
            return top
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return None


def mirror_package(src_root, farm_dir, dotted_name):
    """Symlink <src_root>/Package/Subpackage/python/** into
    <farm_dir>/Package/Subpackage/**, with __init__.py at every level, so it
    becomes importable as Package.Subpackage.* from farm_dir."""
    parts = dotted_name.split(".")
    if len(parts) < 2:
        return False
    package, subpackage = parts[0], parts[1]
    src = os.path.join(src_root, package, subpackage, "python")
    if not os.path.isdir(src):
        return False
    dst = os.path.join(farm_dir, package, subpackage)
    if os.path.isdir(dst):
        return False  # already mirrored, nothing new to add -> real failure
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    def _copy(s, d):
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "__init__.py"), "a").close()
        for entry in sorted(os.listdir(s)):
            sp = os.path.join(s, entry)
            dp = os.path.join(d, entry)
            if os.path.isdir(sp):
                _copy(sp, dp)
            elif entry.endswith(".py"):
                os.symlink(os.path.abspath(sp), dp)

    _copy(src, dst)
    open(os.path.join(farm_dir, package, "__init__.py"), "a").close()
    print(f"  + mirrored {package}/{subpackage} from local checkout", file=sys.stderr)
    return True


def build_dump(src_root, farm_dir, menu, gtemulator, out_file, flat=False):
    driver_path = os.path.join(farm_dir, "_driver.py")
    with open(driver_path, "w") as f:
        f.write(DRIVER_TEMPLATE.format(menu=menu, gtemulator=gtemulator, flat=flat))

    env = dict(os.environ)
    env["DUMP_OUT_FILE"] = os.path.abspath(out_file)
    env["PYTHONPATH"] = farm_dir + os.pathsep + env.get("PYTHONPATH", "")

    for attempt in range(1, MAX_MIRROR_ITERATIONS + 1):
        result = subprocess.run(
            [sys.executable, driver_path], env=env, capture_output=True, text=True,
        )
        if result.returncode == 0 and "DUMP_OK" in result.stdout:
            print(result.stdout.strip())
            return True

        m = re.search(r"ModuleNotFoundError: No module named '([\w.]+)'", result.stderr)
        if not m:
            sys.stderr.write(result.stderr)
            raise RuntimeError("dump failed for a reason other than a missing module (see traceback above)")

        missing = m.group(1)
        if src_root is None:
            raise RuntimeError(
                f"module '{missing}' is not importable and no --src CMSSW checkout was given/found "
                "to mirror it from; either run this inside `cmsenv` or pass --src"
            )
        if not mirror_package(src_root, farm_dir, missing):
            sys.stderr.write(result.stderr)
            raise RuntimeError(
                f"module '{missing}' could not be resolved: not found under "
                f"{src_root}/<Package>/<Subpackage>/python (and not already mirrored either -- "
                "likely needs a real CMSSW release area, not just this git checkout)"
            )

    raise RuntimeError(f"gave up after {MAX_MIRROR_ITERATIONS} mirroring attempts")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default=None, help="CMSSW source checkout to mirror missing pure-python packages from")
    parser.add_argument("--menu", default=DEFAULT_MENU, help="dotted module path of the menu cff to dump")
    parser.add_argument("--gtemulator", default=DEFAULT_GTEMULATOR, help="dotted module path of the GT producer/algo-block-producer cff")
    parser.add_argument("--out", default=None, help="output .py file (default: <last menu component>_fullConfigDump.py)")
    parser.add_argument("--flat", action="store_true", help="inline objects and compute online thresholds (emulator view)")
    parser.add_argument("--keep-farm", action="store_true", help="don't delete the temporary python farm dir on exit")
    args = parser.parse_args()

    out_file = args.out or f"{args.menu.rsplit('.', 1)[-1]}_{'flat' if args.flat else 'full'}ConfigDump.py"
    src_root = args.src or find_default_src_root(__file__)

    farm_dir = tempfile.mkdtemp(prefix="l1tgtmenu_pyfarm_")
    try:
        build_dump(src_root, farm_dir, args.menu, args.gtemulator, out_file, flat=args.flat)
        print(f"wrote {out_file}")
    finally:
        if args.keep_farm:
            print(f"python farm left at {farm_dir}")
        else:
            shutil.rmtree(farm_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
