#!/usr/bin/env python3
"""Self-test of hltPhase2MenuCheck: build a small fake HLT_75e33 tree with one
planted defect per rule and verify that exactly the expected findings come out."""

import json
import os
import shutil
import subprocess
import sys
import tempfile

FILES = {
    "HLT_75e33_cff.py": """import FWCore.ParameterSet.Config as cms
fragment = cms.ProcessFragment("HLT")
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltESPGood_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/eventsetup/hltESPMissing_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLT_Good_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLT_Bad_cfi")
fragment.load("HLTrigger/Configuration/HLT_75e33/paths/HLT_Good_cfi")
""",
    "HLT_75e33/eventsetup/hltESPGood_cfi.py": 'import FWCore.ParameterSet.Config as cms\nhltESPGood = cms.ESProducer("Foo")\n',
    "HLT_75e33/modules/hltA_cfi.py": 'import FWCore.ParameterSet.Config as cms\nhltA = cms.EDProducer("X")\n',
    "HLT_75e33/modules/hltB_cfi.py": 'import FWCore.ParameterSet.Config as cms\nhltB = cms.EDProducer("X")\n',
    "HLT_75e33/modules/hltOrphan_cfi.py": 'import FWCore.ParameterSet.Config as cms\nhltOrphan = cms.EDProducer("X")\n',
    "HLT_75e33/modules/hltStale_cfi.py": 'import FWCore.ParameterSet.Config as cms\nhltStale = cms.EDProducer("X")\n',
    "HLT_75e33/modules/hltC_cfi.py": 'import FWCore.ParameterSet.Config as cms\nhltWrongName = cms.EDFilter("X")\n',
    "HLT_75e33/sequences/HLTGoodSeq_cfi.py": """import FWCore.ParameterSet.Config as cms
from ..modules.hltA_cfi import *
from ..modules.hltB_cfi import *
from RecoTracker.IterativeTracking.Something_cff import *
HLTGoodSeq = cms.Sequence(hltA + hltB + somethingFromReco)
""",
    "HLT_75e33/paths/HLT_Good_cfi.py": """import FWCore.ParameterSet.Config as cms
from ..sequences.HLTGoodSeq_cfi import *
from ..modules.hltC_cfi import *
from ..modules.hltStale_cfi import *
HLT_Good = cms.Path(HLTGoodSeq + hltWrongName)
""",
    "HLT_75e33/paths/HLT_Bad_cfi.py": """import FWCore.ParameterSet.Config as cms
from ..modules.hltA_cfi import *
from ..modules.hltGone_cfi import *
HLT_Bad = cms.Path(hltA + hltNotImported)
""",
    "HLT_75e33/psets/broken_cfi.py": "x = (\n",
}

# (code, file suffix) pairs that must be reported, and nothing else
EXPECTED = {
    ("E001", "psets/broken_cfi.py"),
    ("E002", "paths/HLT_Bad_cfi.py"),          # hltGone_cfi does not exist
    ("E002", "HLT_75e33_cff.py"),              # hltESPMissing_cfi does not exist
    ("E003", "paths/HLT_Bad_cfi.py"),          # hltNotImported
    ("W101", "modules/hltOrphan_cfi.py"),
    ("W101", "psets/broken_cfi.py"),
    ("W102", "paths/HLT_Good_cfi.py"),         # hltStale imported, unused
    ("W103", "modules/hltC_cfi.py"),           # defines hltWrongName
    ("W105", "HLT_75e33_cff.py"),              # HLT_Good loaded twice
}


def main():
    tool = shutil.which("hltPhase2MenuCheck") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "hltPhase2MenuCheck")
    tmp = tempfile.mkdtemp()
    try:
        py = os.path.join(tmp, "HLTrigger", "Configuration", "python")
        for sub in ("", "modules", "sequences", "paths", "eventsetup", "psets"):
            os.makedirs(os.path.join(py, "HLT_75e33", sub), exist_ok=True)
            open(os.path.join(py, "HLT_75e33", sub, "__init__.py"), "w").close()
        for rel, text in FILES.items():
            with open(os.path.join(py, rel), "w") as f:
                f.write(text)
        res = subprocess.run([sys.executable, tool, "--python-dir", py, "--json", "-"],
                             stdout=subprocess.PIPE, universal_newlines=True)
        findings = json.loads(res.stdout)["findings"]
        got = set()
        for f in findings:
            for code, suffix in EXPECTED:
                if f["code"] == code and f["file"].endswith(suffix):
                    got.add((code, suffix))
                    break
            else:
                got.add((f["code"], f["file"]))
        missing, extra = EXPECTED - got, got - EXPECTED
        for m in sorted(missing):
            print("MISSING expected finding: %s %s" % m)
        for e in sorted(extra):
            print("UNEXPECTED finding: %s %s" % e)
        if res.returncode != 1:
            print("UNEXPECTED exit code %d (errors present, expected 1)" % res.returncode)
            return 1
        if missing or extra:
            return 1
        print("hltPhase2MenuCheck self-test passed (%d findings)" % len(findings))
        return 0
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    sys.exit(main())
