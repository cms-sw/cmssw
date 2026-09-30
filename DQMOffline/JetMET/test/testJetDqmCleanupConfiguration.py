#!/usr/bin/env python3
"""Check era activation and clone isolation against the declared jet selection.

Each scenario runs in a fresh Python interpreter because CMSSW modifiers keep
process-local state. These are configuration checks, including an isolated
phase2_hgcal activation; they do not execute Phase-2 reconstruction or events.
"""

import json
from pathlib import Path
import subprocess
import sys


SCENARIOS = ("default", "Run3_2026", "phase2_hgcal", "pp_on_AA", "cosmics")
MANIFEST = Path(__file__).resolve().parents[1] / "doc" / "jetDqmCleanupSelection.json"


def check_scenario(scenario):
    import FWCore.ParameterSet.Config as cms
    from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal

    modifiers = []
    if scenario == "Run3_2026":
        from Configuration.Eras.Era_Run3_2026_cff import Run3_2026

        modifiers.append(Run3_2026)
    elif scenario == "phase2_hgcal":
        modifiers.append(phase2_hgcal)
    elif scenario == "pp_on_AA":
        from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA

        modifiers.append(pp_on_AA)
    process = cms.Process("JETCONFIGTEST", *modifiers)
    process.load("DQMOffline.JetMET.jetAnalyzer_cff")
    process.load("DQMOffline.JetMET.dataCertificationJetMET_cfi")
    if scenario == "cosmics":
        process.load("DQMOffline.JetMET.jetMETDQMOfflineSourceCosmic_cff")

    manifest = json.loads(MANIFEST.read_text())
    targets = {tab: values["python_instance"] for tab, values in manifest["configuration_targets"].items()}
    expected = {label: set() for label in targets.values()}
    selected_keys = set()
    conditional_keys = set()
    hgcal_active = scenario == "phase2_hgcal"
    for row in manifest["rows"]:
        key = (row["tab"], row["name"])
        assert key not in selected_keys, f"Duplicate manifest selection: {key}"
        selected_keys.add(key)
        assert row["decision"] in ("Discard", "Discard/replace for HGCAL"), row
        conditional = row["decision"] == "Discard/replace for HGCAL"
        if conditional:
            conditional_keys.add(key)
        if hgcal_active or (not conditional and "HGCAL" not in row["note_flags"]):
            expected[targets[row["tab"]]].add(row["name"])
    assert len(conditional_keys) == 5
    assert all(tab == "PUPPI_JET" for tab, _ in conditional_keys)
    assert bool(phase2_hgcal._isChosen()) == hgcal_active

    actual = {}
    controls = {}
    for label, module in process.producers_().items():
        if module.type_() != "JetAnalyzer":
            continue
        values = list(module.disabledMEs.value())
        assert len(values) == len(set(values)), f"Duplicate disabled names on {label}"
        actual[label] = values
        if label in expected:
            assert set(values) == expected[label], (
                scenario, label, "missing", sorted(expected[label] - set(values)),
                "unexpected", sorted(set(values) - expected[label]),
            )
        else:
            assert not values, f"Cleanup propagated to control clone {label}: {values}"
            controls[label] = values
    assert set(expected).issubset(actual), (scenario, "Selected instances missing")
    assert "jetDQMAnalyzerAk4PFCHSCleaned" in controls
    assert "jetDQMAnalyzerAk4PFUncleaned" in controls
    assert not process.jetDQMAnalyzerAk4CaloUncleaned.disabledMEs.value()
    assert process.dataCertificationJetMET.JetTypeRECO.value() == "ak4PFJetsCHS"

    if scenario == "cosmics":
        sequence = set(process.jetDQMAnalyzerSequenceCosmics.moduleNames())
        assert sequence == {"jetDQMAnalyzerAk4CaloUncleaned"}
        assert process.jetDQMAnalyzerAk4CaloUncleaned.runcosmics.value()
    else:
        sequence = set(process.jetDQMAnalyzerSequence.moduleNames())
        if scenario == "pp_on_AA":
            assert not sequence.intersection(expected), "pp_on_AA should schedule the HI jet sequence"
        else:
            assert set(expected).issubset(sequence)
    return {
        "scenario": scenario,
        "phase2_hgcal_chosen": hgcal_active,
        "selected_lists": {label: actual[label] for label in expected},
        "selected_name_count": sum(len(expected[label]) for label in expected),
        "control_clones_with_empty_lists": sorted(controls),
        "active_jet_sequence_module_names": sorted(sequence),
        "default_harvester_RECO_collection": process.dataCertificationJetMET.JetTypeRECO.value(),
        "scope": "Configuration only; no event or Phase-2 physics validation",
    }


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--scenario":
        scenario = sys.argv[2]
        if scenario not in SCENARIOS:
            raise ValueError("Unknown scenario: " + scenario)
        print(json.dumps(check_scenario(scenario), sort_keys=True))
        return
    results = []
    for scenario in SCENARIOS:
        output = subprocess.check_output(
            [sys.executable, str(Path(__file__).resolve()), "--scenario", scenario],
            text=True,
        )
        results.append(json.loads(output))
    print(json.dumps({"passed": True, "scenarios": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
