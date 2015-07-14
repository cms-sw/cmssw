import FWCore.ParameterSet.Config as cms
import copy

from Configuration.EventContent.EventContent_cff import FEVTEventContent

OutALCARECOHotline_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring(
            "pathHotlineSkimSingleMuon",
            "pathHotlineSkimDoubleMuon",
            "pathHotlineSkimTripleMuon",
            "pathHotlineSkimSingleElectron",
            "pathHotlineSkimDoubleElectron",
            "pathHotlineSkimTripleElectron",
            "pathHotlineSkimSinglePhoton",
            "pathHotlineSkimDoublePhoton",
            "pathHotlineSkimTriplePhoton",
            "pathHotlineSkimSingleJet",
            "pathHotlineSkimDoubleJet",
            "pathHotlineSkimMultiJet",
            "pathHotlineSkimHT",
            "pathHotlineSkimMassiveDimuon",
            "pathHotlineSkimMassiveDielectron",
            "pathHotlineSkimMassiveEMu",
            "pathHotlineSkimPFMET",
            "pathHotlineSkimCaloMET",
            "pathHotlineSkimCondMET",
        ),
    ),
    outputCommands = copy.deepcopy(FEVTEventContent.outputCommands)
)

while 'drop *' in OutALCARECOHotline_noDrop.outputCommands: OutALCARECOHotline_noDrop.outputCommands.remove('drop *')

import copy
OutALCARECOHotline = copy.deepcopy(OutALCARECOHotline_noDrop)
OutALCARECOHotline.outputCommands.insert(0, "drop *")
