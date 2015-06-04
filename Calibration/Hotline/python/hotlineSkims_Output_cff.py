import FWCore.ParameterSet.Config as cms

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
            "pathHotlineSkimMassiveEMu"
        ),
    ),
    outputCommands = cms.untracked.vstring(
        'keep *'
        )
)
