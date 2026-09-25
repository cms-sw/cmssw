"""Each era runs in a separate process because modifiers are global objects."""
import json
from pathlib import Path
import subprocess
import sys


def check(era):
    import FWCore.ParameterSet.Config as cms
    from Configuration.StandardSequences.Eras import eras
    modifiers = [getattr(eras, era.removesuffix("_FastSim"))]
    if era.endswith("_FastSim"):
        modifiers.append(eras.fastSim)
    process = cms.Process("TEST", *modifiers)
    process.load("PhysicsTools.NanoAOD.nano_cff")
    from PhysicsTools.NanoAOD.jetsAK4_Puppi_cff import nanoAOD_refineFastSim_puppiJet
    process = nanoAOD_refineFastSim_puppiJet(process)
    variables = process.jetPuppiTable.variables
    selected = era == "Run3_2024_FastSim"
    model = "fastSimPuppiJetRefineNN_Run2024_UnitaryShapeBest150_20260918.onnx" if selected else "fastSimPuppiJetRefineNN_31July2025.onnx"
    assert process.puppiJetRefineNN.weightFile.value().endswith(model)
    assert hasattr(process.processRefinedJets, "minGenJetPt") == selected
    assert ("fastSimFinal_" in variables.btagDeepFlavB.expr.value()) == selected
    assert hasattr(process.processRefinedJets, "taggerNames") == selected
    assert ("isAvailable" in process.puppiJetRefineNN.variables[0].expr.value()) == selected
    if selected:
        assert process.processRefinedJets.minGenJetPt.value() == 10.
        assert variables.pt.expr.value() == "userFloat('pt_final')"
        features = ("btagDeepFlavB", "btagDeepFlavCvB", "btagDeepFlavCvL", "btagDeepFlavQG",
                    "btagUParTAK4B", "btagUParTAK4CvB", "btagUParTAK4CvL", "btagUParTAK4QvG")
        assert tuple(process.processRefinedJets.taggerNames) == features
        for i, feature in enumerate(features):
            raw = getattr(variables, feature+"_unrefined").expr.value()
            assert process.processRefinedJets.rawTaggerExpressions[i] == raw
            assert getattr(variables, feature).expr.value() == "userFloat('fastSimFinal_"+feature+"')"
    if "FastSim" in era:
        assert process.linkedObjects.jets.value() == "finalJetsPuppiSorted"
        assert process.jetPuppiTable.src.value() == "linkedObjects:jets"
        assert process.jetMCTable.src.value() == "linkedObjects:jets"
        assert process.puppiMetTable.src.value() == "processRefinedJets:Refined"
    else:
        assert process.linkedObjects.jets.value() == "finalJetsPuppi"
        assert not hasattr(variables, "pt_unrefined")
        assert process.puppiMetTable.src.value() != "processRefinedJets:Refined"
    print(json.dumps({"era":era,"pass":True,"payload":model}))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        check(sys.argv[1])
    else:
        from Configuration.StandardSequences.Eras import eras
        era_names = ["Run3_FastSim", "Run3_2023_FastSim", "Run3_2024_FastSim", "Run3_2025_FastSim",
                     "Run3_2024", "Run3"]
        if hasattr(eras, "Run3_2026"):
            era_names.append("Run3_2026_FastSim")
        for era in era_names:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), era], check=True)
