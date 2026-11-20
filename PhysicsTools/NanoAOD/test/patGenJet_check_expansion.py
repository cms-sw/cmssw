#!/usr/bin/env python3
"""Configuration comparisons on STANDARD cmsDriver configurations (MiniAOD step and NANO step, MC, Run3_2026;
cmsDriver.py --no_exec works offline: no network is needed to generate them).

import check: importing the package's python modules (tool, customisations, cffs: a Python import, no process.load) changes nothing:
    process.dumpPython() of the MiniAOD and NANO configuration is identical with and without the imports.
customisation check: without a call to a customisation: identical (same statement as the import check: the customisation is only a function);
    with customisePatGenJetsNano(process): every module, PSet, output module and task of the standard configuration is unchanged
    except the tasks the new tables are added to; the ONLY new modules are patGenJetAK4Table, patGenJetAK8Table,
    patSubGenJetAK8Table (+ patGenJetTablesTask);
    with customisePatGenJetsMiniAOD(process): the only new modules are those of patGenJetsTask; the output modules are
    unchanged; with keepInOutput=True the only change is the two `keep` lines in the MiniAOD output module.

Usage (under cmsenv): python3 patGenJet_check_expansion.py [work_dir]      Exit code 0 if all hold.
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
work = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="patgen_expansion_")
os.makedirs(work, exist_ok=True)

DRIVER = {
    "nano": ["cmsDriver.py", "nano", "--step", "NANO", "--mc", "--eventcontent", "NANOAODSIM", "--datatier", "NANOAODSIM",
             "--era", "Run3_2026", "--conditions", "auto:phase1_2026_realistic", "--filein", "file:dummy.root",
             "--fileout", "file:out.root", "-n", "1", "--no_exec", "--python_filename", "nano_cfg.py"],
    "mini": ["cmsDriver.py", "mini", "--step", "PAT", "--mc", "--eventcontent", "MINIAODSIM", "--datatier", "MINIAODSIM",
             "--era", "Run3_2026", "--conditions", "auto:phase1_2026_realistic", "--filein", "file:dummy.root",
             "--fileout", "file:out.root", "-n", "1", "--no_exec", "--python_filename", "mini_cfg.py"],
}
for kind, cmd in DRIVER.items():
    if not os.path.exists(os.path.join(work, kind + "_cfg.py")):
        subprocess.run(cmd, cwd=work, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# one python process per variant, so that nothing leaks between them
RUNNER = r'''
import sys, json, importlib.util
variant, cfg = sys.argv[1], sys.argv[2]
if variant != "plain":
    import PhysicsTools.PatAlgos.tools.patGenJetTools, PhysicsTools.PatAlgos.slimming.customisePatGenJetsMiniAOD
    import PhysicsTools.NanoAOD.customisePatGenJetsNano
    import PhysicsTools.PatAlgos.producersLayer1.patGenJets_cff, PhysicsTools.NanoAOD.patGenJetTables_cff, PhysicsTools.PatAlgos.tools.patGenJetHelpers
spec = importlib.util.spec_from_file_location("cfg", cfg); m = importlib.util.module_from_spec(spec); sys.argv = [cfg]; spec.loader.exec_module(m)
process = m.process
if variant == "nano":
    from PhysicsTools.NanoAOD.customisePatGenJetsNano import customisePatGenJetsNano
    process = customisePatGenJetsNano(process)
elif variant == "mini":
    from PhysicsTools.PatAlgos.slimming.customisePatGenJetsMiniAOD import customisePatGenJetsMiniAOD
    process = customisePatGenJetsMiniAOD(process)
elif variant == "mini_keep":
    from PhysicsTools.PatAlgos.slimming.customisePatGenJetsMiniAOD import customisePatGenJetsMiniAOD
    process = customisePatGenJetsMiniAOD(process, keepInOutput=True)
out = {"full": process.dumpPython()}
for kind, d in (("producer", process.producers_()), ("filter", process.filters_()), ("analyzer", process.analyzers_()),
                ("output", process.outputModules_()), ("task", process.tasks_()), ("sequence", process.sequences_()),
                ("path", process.paths_()), ("endpath", process.endpaths_()), ("pset", process.psets_()),
                ("service", process.services_()), ("essource", process.es_sources_()), ("esproducer", process.es_producers_())):
    for label, obj in d.items():
        out[kind + ":" + label] = obj.dumpPython()
for label, obj in process.outputModules_().items():
    out["cmds:" + label] = list(obj.outputCommands)
out["schedule"] = str(process.schedule)
json.dump(out, sys.stdout)
'''
open(os.path.join(work, "runner.py"), "w").write(RUNNER)

import json


def run(variant, kind):
    r = subprocess.run([sys.executable, os.path.join(work, "runner.py"), variant, os.path.join(work, kind + "_cfg.py")],
                       capture_output=True, text=True, cwd=work)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-4000:])
        raise SystemExit("variant %s %s failed" % (variant, kind))
    return json.loads(r.stdout[r.stdout.index("{"):])


bad = []
def check(cond, msg):
    if not cond:
        bad.append(msg)

for kind in ("nano", "mini"):
    plain = run("plain", kind)
    imported = run("imports", kind)
    check(plain == imported, "%s: the dump differs after importing the package python modules" % kind)
    print("%s: %d configuration objects; identical with the package modules imported: %s" % (kind, len(plain), plain == imported))

# ---- customisation check: NANO
plain, cust = run("plain", "nano"), run("nano", "nano")
new_keys = sorted(set(cust) - set(plain))
changed = sorted(k for k in plain if k in cust and plain[k] != cust[k])
print("nano customisation: new objects", new_keys)
print("nano customisation: changed objects", changed)
check(sorted(k for k in new_keys if k.startswith(("producer:", "filter:", "analyzer:"))) ==
      ["producer:patGenJetAK4Table", "producer:patGenJetAK8Table", "producer:patSubGenJetAK8Table"], "nano: new modules are %s" % new_keys)
check(set(k for k in new_keys if k.startswith("task:")) == {"task:patGenJetTablesTask"}, "nano: new tasks")
check(not [k for k in new_keys if not k.startswith(("producer:", "task:"))], "nano: other new objects %s" % new_keys)
check(set(changed) <= {"full", "task:nanoTableTaskFS", "sequence:nanoSequenceFS", "sequence:nanoSequenceMC", "path:nanoAOD_step"},
      "nano: unexpected changes %s" % changed)
check("patGenJetTablesTask" in cust["task:nanoTableTaskFS"] and "patGenJetTablesTask" not in plain["task:nanoTableTaskFS"],
      "nano: tables not in nanoTableTaskFS")
# the part of nanoTableTaskFS that is not ours is unchanged
check(cust["task:nanoTableTaskFS"].replace("process.patGenJetTablesTask, ", "") == plain["task:nanoTableTaskFS"],
      "nano: nanoTableTaskFS differs by more than our task")
check(plain["output:NANOAODSIMoutput"] == cust["output:NANOAODSIMoutput"], "nano: output module changed")

# ---- customisation check: MiniAOD
plain, cust, keep = run("plain", "mini"), run("mini", "mini"), run("mini_keep", "mini")
new_keys = sorted(set(cust) - set(plain))
changed = sorted(k for k in plain if k in cust and plain[k] != cust[k])
print("mini customisation: %d new objects, changed objects %s" % (len(new_keys), changed))
expected = ["NjettinessAK8PuppiForPatGenJets", "NjettinessAK8SubjetsForPatGenJets", "ak8PFJetsPuppiSoftDropMassForPatGenJets", "nb1AK8PuppiSoftDropForPatGenJets", "nb1AK8PuppiSoftDropSubjetsForPatGenJets",
            "nb2AK8PuppiSoftDropForPatGenJets", "nb2AK8PuppiSoftDropSubjetsForPatGenJets", "packedPatGenJetsAK8", "patGenJetsAK4",
            "patGenJetsAK8", "patGenJetsAK8SoftDropSubjets", "selectedPatGenJetsAK8"]
mods = sorted(k.split(":")[1] for k in new_keys if k.startswith(("producer:", "filter:", "analyzer:")))
check(mods == sorted(expected), "mini: new modules %s" % mods)
check(set(k for k in new_keys if k.startswith("task:")) == {"task:patGenJetsTask"}, "mini: new tasks")
check(not [k for k in new_keys if not k.startswith(("producer:", "filter:", "task:"))], "mini: other new objects %s" % new_keys)
check(set(changed) <= {"full", "task:patAlgosToolsTask"}, "mini: unexpected changes %s" % changed)
check("patGenJetsTask" in cust["task:patAlgosToolsTask"], "mini: task not added to patAlgosToolsTask")
out = "output:MINIAODSIMoutput"
check(plain[out] == cust[out], "mini: output module changed without keepInOutput")
added = [c for c in keep["cmds:MINIAODSIMoutput"] if c not in plain["cmds:MINIAODSIMoutput"]]
print("mini keepInOutput: added output commands:", added)
check(added == ["keep *_patGenJetsAK4_*_*", "keep *_packedPatGenJetsAK8_*_*"], "mini keep: added commands %s" % added)
check([c for c in plain["cmds:MINIAODSIMoutput"] if c not in keep["cmds:MINIAODSIMoutput"]] == [], "mini keep: commands removed")
check(cust["cmds:MINIAODSIMoutput"] == plain["cmds:MINIAODSIMoutput"], "mini: output commands changed without keepInOutput")
check(sorted(k for k in keep if k not in plain) == sorted(k for k in cust if k not in plain), "mini keep: different new objects")

if bad:
    print("FAILED:")
    for b in bad:
        print("  ", b)
    sys.exit(1)
print("check_expansion: OK")
