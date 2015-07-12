import FWCore.ParameterSet.Config as cms
import copy

from Configuration.EventContent.EventContent_cff import FEVTEventContent

OutALCARECOHotline = cms.PSet(
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

while 'drop *' in OutALCARECOHotline.outputCommands: OutALCARECOHotline.outputCommands.remove('drop *')
