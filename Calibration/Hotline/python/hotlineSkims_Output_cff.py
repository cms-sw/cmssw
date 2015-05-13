import FWCore.ParameterSet.Config as cms

OutALCARECOHotline = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring(
            "hotlineSkimSingleMuon",
            "hotlineSkimDoubleMuon",
            "hotlineSkimTripleMuon",
            "hotlineSkimSingleElectron",
            "hotlineSkimDoubleElectron",
            "hotlineSkimTripleElectron",
            "hotlineSkimSinglePhoton",
            "hotlineSkimDoublePhoton",
            "hotlineSkimTriplePhoton",
            "hotlineSkimSingleJet",
            "hotlineSkimDoubleJet",
            "hotlineSkimMultiJet",
            "hotlineSkimHT",
            "hotlineSkimMassiveDimuon",
            "hotlineSkimMassiveDielectron",
            "hotlineSkimMassiveEMu"
        ),
    ),
    outputCommands = cms.untracked.vstring('keep *')
)
