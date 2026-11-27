#!/usr/bin/env python3
"""The columns of the PatGen* tables against the Jet/FatJet/SubJet columns of the running NanoAOD process.

Usage (under cmsenv):  python3 patGenJet_check_columns.py [nano_cfg.py]
  nano_cfg.py: a cmsDriver NANO MC configuration (cmsDriver.py ... --step NANO --mc --no_exec, works offline).
  Without it the process is built in Python (era Run3_2026, nano_cff, nanoAOD_customizeCommon): same columns.

Reference = variables + externalVariables of  jetPuppiTable+jetMCTable (Jet), fatJetTable+fatJetMCTable (FatJet),
subJetTable+subjetMCTable (SubJet), read after nanoAOD_customizeCommon and BEFORE customisePatGenJetsNano.
Checks, for each pair (Jet-PatGenJet, FatJet-PatGenJetAK8, SubJet-PatSubGenJetAK8):
  - every reference column whose expression reads bDiscriminator (taggers, regressions) and every column in
    patGenJetTables_cff.documentedChanges[...]['removed'] is ABSENT from the new table
  - every other reference column exists in the new table with the same type, precision and expression, except the
    documented changes (patGenJetTables_cff.documentedChanges): pt precision -1; the FatJet isPFJet() guards removed
  - the new table has exactly the kept reference columns plus the documented additions (names, types)
  - src, cut, name, module label ending in Table; the DQM entries exist when the process has nanoDQM
  - nothing else changed in the process (every other module equal before/after)
Exit code 0 if everything holds.
"""
import importlib.util
import re
import sys

import FWCore.ParameterSet.Config as cms


# columns that are not written for generator jets (constant, flavour and gen-jet index columns, 39); every other non-tagger column is kept
REMOVED = {
    "PatGenJet": ["rawFactor", "area", "hfEmEF", "hfHEF", "hfsigmaEtaEta", "hfsigmaPhiPhi", "hfcentralEtaStripSize",
                  "hfadjacentEtaStripsSize", "puIdDisc", "muonSubtrFactor", "muonSubtrDeltaEta", "muonSubtrDeltaPhi",
                  "nElectrons", "nMuons", "nSVs", "electronIdx1", "electronIdx2", "muonIdx1", "muonIdx2", "svIdx1", "svIdx2",
                  "partonFlavour", "hadronFlavour", "genJetIdx"],
    "PatGenJetAK8": ["rawFactor", "area", "hfEmEF", "hfHEF", "lsf3", "electronIdx3SJ", "muonIdx3SJ",
                     "hadronFlavour", "genJetAK8Idx"],
    "PatSubGenJetAK8": ["rawFactor", "area", "nBHadrons", "nCHadrons", "hadronFlavour", "subGenJetAK8Idx"],
}


def load_process(cfg):
    if cfg:
        spec = importlib.util.spec_from_file_location("nanocfg", cfg)
        mod = importlib.util.module_from_spec(spec)
        sys.argv = [cfg]
        spec.loader.exec_module(mod)
        return mod.process
    from Configuration.Eras.Era_Run3_2026_cff import Run3_2026
    process = cms.Process("X", Run3_2026)
    process.load("PhysicsTools.NanoAOD.nano_cff")
    from PhysicsTools.NanoAOD.nano_cff import nanoAOD_customizeCommon
    return nanoAOD_customizeCommon(process)


def columns(*tables):
    cols = {}
    for t in tables:
        for blk in ("variables", "externalVariables"):
            if not hasattr(t, blk):
                continue
            ps = getattr(t, blk)
            for n in ps.parameterNames_():
                p = getattr(ps, n)
                assert n not in cols, "duplicate column %s" % n
                cols[n] = dict(kind="Var" if blk == "variables" else "ExtVar", type=p.type.value(),
                               precision=p.precision.value() if hasattr(p, "precision") else -1,
                               expr=p.expr.value() if hasattr(p, "expr") else p.src.configValue())
    return cols


def main():
    cfg = sys.argv[1] if len(sys.argv) > 1 else None
    process = load_process(cfg)
    nanoDQM = False
    try:
        process.load("PhysicsTools.NanoAOD.nanoDQM_cff")
        nanoDQM = hasattr(process, "nanoDQM")
    except Exception as e:  # DQM packages missing: skip that check
        print("note: nanoDQM not loadable (%s)" % e)
    ref = {
        "Jet": columns(process.jetPuppiTable, process.jetMCTable),
        "FatJet": columns(process.fatJetTable, process.fatJetMCTable),
        "SubJet": columns(process.subJetTable, process.subjetMCTable),
    }
    before = {l: m.dumpPython() for d in (process.producers_(), process.filters_(), process.analyzers_()) for l, m in d.items()}
    nTasksBefore = set(process.tasks_().keys())

    from PhysicsTools.NanoAOD.customisePatGenJetsNano import customisePatGenJetsNano
    from PhysicsTools.NanoAOD.patGenJetTables_cff import documentedChanges, TAGGER_EXPR
    process = customisePatGenJetsNano(process)

    pairs = [("Jet", "patGenJetAK4Table", "PatGenJet", "patGenJetsAK4", "pt > 10"),
             ("FatJet", "patGenJetAK8Table", "PatGenJetAK8", "packedPatGenJetsAK8", "pt > 100."),
             ("SubJet", "patSubGenJetAK8Table", "PatSubGenJetAK8", "packedPatGenJetsAK8:SubJets", "")]
    bad = []
    for old, label, name, src, cut in pairs:
        mod = getattr(process, label)
        assert label.endswith("Table")
        if mod.name.value() != name or mod.src.configValue() != src or mod.cut.value() != cut:
            bad.append("%s: name/src/cut = %s %s '%s'" % (label, mod.name.value(), mod.src.configValue(), mod.cut.value()))
        new = columns(mod)
        doc = documentedChanges[name]
        if sorted(doc["removed"]) != sorted(REMOVED[name]):
            bad.append("%s: removed columns %s, expected %s" % (name, sorted(doc["removed"]), sorted(REMOVED[name])))
        unknown = sorted(set(doc["removed"]) - set(ref[old]))
        if unknown:
            bad.append("%s: removed columns %s are not in the reference" % (name, unknown))
        n_same = 0
        kept = {}
        n_tagger = n_removed = 0
        for c, r in ref[old].items():
            tagger = TAGGER_EXPR in r["expr"]
            if tagger or c in doc["removed"]:
                if tagger:
                    n_tagger += 1
                else:
                    n_removed += 1
                if c in new:
                    bad.append("%s: column %s must be absent (%s)" % (name, c, "tagger/regression" if tagger else "removed"))
                continue
            kept[c] = r
            if c not in new:
                bad.append("%s: column %s missing" % (name, c))
                continue
            n = new[c]
            if n["type"] != r["type"]:
                bad.append("%s.%s: type %s != %s" % (name, c, n["type"], r["type"]))
            if c in doc["precision"]:
                if n["precision"] != -1 or n["expr"] != r["expr"]:
                    bad.append("%s.%s: precision/expr %s" % (name, c, n))
                continue
            if c in doc["unguarded"]:
                m = re.match(r"^\?isPFJet\(\)\?(.*):-1$", r["expr"])
                if not (m and n["expr"] == m.group(1) and n["precision"] == r["precision"]):
                    bad.append("%s.%s: unguarded expression is %s (reference %s)" % (name, c, n["expr"], r["expr"]))
                continue
            if n["precision"] != r["precision"] or n["expr"] != r["expr"]:
                # the only other allowed rewrite: hasUserFloat guard of a packer-only label
                if not (n["expr"].startswith("?hasUserFloat(") and r["expr"] in n["expr"]):
                    bad.append("%s.%s: differs: %s vs %s" % (name, c, n, r))
                    continue
            n_same += 1
        extra = sorted(set(new) - set(kept))
        if extra != sorted(doc["added"]):
            bad.append("%s: added columns %s, documented %s" % (name, extra, sorted(doc["added"])))
        if any(TAGGER_EXPR in v["expr"] for v in new.values()):
            bad.append("%s: a column reads %s" % (name, TAGGER_EXPR))
        print("%-16s reference %3d columns: tagger/regression %2d and removed %2d absent; kept %2d (unchanged %2d, "
              "documented changes %2d); added %2d; total %2d  [%s]"
              % (name, len(ref[old]), n_tagger, n_removed, len(kept), n_same, len(kept) - n_same, len(extra), len(new),
                 ", ".join(extra)))
        if nanoDQM and not hasattr(process.nanoDQM.vplots, name):
            bad.append("%s: no nanoDQM entry" % name)
        elif nanoDQM:
            plotted = set(p.name.value() for p in getattr(process.nanoDQM.vplots, name).plots)
            unknown = [p for p in plotted if not p.startswith("_") and p not in new]
            if unknown:
                bad.append("%s: DQM plots of unknown columns %s" % (name, unknown))
    # only our modules and tasks were added
    after = {l: m.dumpPython() for d in (process.producers_(), process.filters_(), process.analyzers_()) for l, m in d.items()}
    changed = [l for l in before if l in after and before[l] != after[l] and l != "nanoDQM"]
    removed = [l for l in before if l not in after]
    added = sorted(set(after) - set(before))
    if changed or removed:
        bad.append("existing modules changed %s removed %s" % (changed, removed))
    if added != ["patGenJetAK4Table", "patGenJetAK8Table", "patSubGenJetAK8Table"]:
        bad.append("modules added: %s" % added)
    print("modules added:", added, " tasks added:", sorted(set(process.tasks_().keys()) - nTasksBefore))
    if not nanoDQM:
        print("note: no nanoDQM in the process, DQM entries not checked")
    if bad:
        print("FAILED:")
        for b in bad:
            print("  ", b)
        return 1
    print("check_columns: OK")
    return 0


sys.exit(main())
