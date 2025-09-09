#!/usr/bin/env python3
import importlib.util
import os
import tempfile, shutil
import subprocess
import sys

from HLTrigger.Configuration.Tools.confdb import HLTProcess
import FWCore.ParameterSet.Config as cms

class _DummyConfig:
    """Minimal config to drive HLTProcess.overrideOutput()."""
    def __init__(self, output):
        self.hilton   = False
        self.fragment = False
        self.output   = output  # "minimal", "all", or "full"

def _load_menu(menu_type):
    """Generate a Run3 HLT menu with cmsDriver and load the process object."""

    with tempfile.TemporaryDirectory() as tmpdir:
        menu_path = os.path.join(tmpdir, "myMenu.py")

        # Run cmsDriver
        cmsdriver_cmd = [
            "cmsDriver.py",
            "TEST",
            "-s", f"L1REPACK:uGT,HLT:{menu_type}",
            "--data",
            "--scenario=pp",
            "-n", "1",
            "--conditions", f"auto:run3_hlt_{menu_type}",
            "--datatier", "RAW",
            "--eventcontent", "RAW",
            "--era", "Run3",
            "--process", "reHLT",
            "--no_exec",
            "--python_filename", menu_path,
        ]

        subprocess.run(cmsdriver_cmd, check=True)

        # Import the generated configuration
        spec = importlib.util.spec_from_file_location("myMenu", menu_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        return mod.process

def _prune_all_outputs(process):
    """Remove ALL existing output modules and any EndPaths that carry them (and unschedule them)."""
    # Collect current output modules
    outputs = list(process.outputModules_())
    if not outputs:
        return

    # Identify EndPaths that contain those outputs
    endpaths_to_remove = []
    try:
        ep_items = process.endpaths_().items()   # available in CMSSW python API
    except Exception:
        ep_items = []
    for ep_name, ep in ep_items:
        try:
            if any(lbl in ep.moduleNames() for lbl in outputs):
                endpaths_to_remove.append(ep_name)
        except Exception:
            pass

    # Unschedule those EndPaths first (if schedule exists)
    if hasattr(process, "schedule") and process.schedule is not None:
        for ep_name in endpaths_to_remove:
            ep = getattr(process, ep_name, None)
            if ep is not None:
                try:
                    process.schedule.remove(ep)
                except Exception:
                    pass

    # Drop EndPaths
    for ep_name in endpaths_to_remove:
        if hasattr(process, ep_name):
            delattr(process, ep_name)

    # Drop the output modules
    for lbl in outputs:
        if hasattr(process, lbl):
            delattr(process, lbl)

def _build_hltprocess_from(process, output_mode: str, input_file: str) -> HLTProcess:
    """Seed an HLTProcess with the (possibly pruned) process text."""
    cfg = _DummyConfig(output=output_mode)
    hlt = HLTProcess.__new__(HLTProcess)  # bypass __init__ (no ConfDB query)
    hlt.config  = cfg
    hlt.config.parent = None
    hlt.config.input = input_file
    hlt.config.emulator = "uGT"
    hlt.data    = process.dumpPython()
    hlt.parent  = []
    hlt.options = {k: [] for k in
                   ['essources','esmodules','modules','sequences','services','paths','psets','blocks']}
    hlt.labels  = {'process': 'process', 'dict': 'process.__dict__'}
    return hlt

def _get_default_input():
    # Path to the filelist
    cmssw_base = os.environ["CMSSW_BASE"]
    filelist = os.path.join(cmssw_base,
                            "src/HLTrigger/Configuration/test/testAccessToEDMInputsOfHLTTests_filelist.txt")

    # Use pure Python (no external grep/tail needed)
    infile = None
    with open(filelist) as f:
        lines = [l.strip() for l in f if "/Run20" in l]
        if lines:
            infile = lines[-1]  # tail -1 equivalent

    if infile is None:
        raise RuntimeError(f"No matching input file found in {filelist}")
    return infile

def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python3 testHLTOutput.py [minimal|all|full] [GRun|HIon|PIon|...] [input_file.root (optional)]")
        sys.exit(1)

    mode = sys.argv[1]
    menu_type = sys.argv[2]
    infile = sys.argv[3] if len(sys.argv) == 4 else _get_default_input()

    if mode not in ("minimal", "all", "full"):
        print(f"Do not understand mode '{mode}'")
        sys.exit(1)

    # --- Nice printouts ---
    print("=" * 80)
    print("Test HLT configuration outputs")
    print("-" * 80)
    print(f" Mode       : {mode}")
    print(f" Menu type  : {menu_type}")
    print(f" Input file : {infile}")
    print("=" * 80)

    # 1) Load the real menu
    process = _load_menu(menu_type)

    # 2) For minimal/full, prune existing outputs so we end up with ONLY the requested one
    if mode in ("minimal", "full"):
        _prune_all_outputs(process)
        # sanity: should be empty now
        assert len(process.outputModules_()) == 0, "Output pruning failed: outputs still present"

    # 3) Wrap in HLTProcess and apply overrideOutput()
    hlt = _build_hltprocess_from(process, mode, infile)
    hlt.overrideOutput()
    hlt.build_source()
        
    # 4) Make the job runnable without input files
    hlt.data += """
# --- test harness tweaks ---
%(process)s.options.wantSummary = cms.untracked.bool(False)
"""

    # 5) Finalize substitutions and write cfg
    cfg_text = hlt.dump()
    cfg_path = f"override_{mode}_cfg.py"
    with open(cfg_path, "w") as f:
        f.write(cfg_text)
    print(f"[ok] wrote {cfg_path}")

    # Optional: quick check of outputs in the final text (informational)
    if mode in ("minimal", "full"):
        expect = "hltOutputMinimal" if mode == "minimal" else "hltOutputFull"
        if expect not in cfg_text:
            print(f"[warn] expected {expect} not found in generated cfg")

    # 6) Run cmsRun
    print(f"[run] cmsRun -j job_{mode}.xml {cfg_path}")
    ret = subprocess.run(["cmsRun", "-j", f"job_{mode}.xml", cfg_path])
    sys.exit(ret.returncode)

if __name__ == "__main__":
    main()
