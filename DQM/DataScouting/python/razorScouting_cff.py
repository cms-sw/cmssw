import FWCore.ParameterSet.Config as cms

#build the hemispheres from our jets
scoutingRHemisphere = cms.EDFilter("HLTRHemisphere",
    acceptNJ = cms.bool(True),
    maxEta = cms.double(3.0),
    inputTag = cms.InputTag("hltCaloJetIDPassed"),
    maxMuonEta = cms.double(2.1),
    muonTag = cms.InputTag(""),
    minJetPt = cms.double(30.0),
    doMuonCorrection = cms.bool(False),
    maxNJ = cms.int32(14)
)

scoutingRazorVariables = cms.EDProducer("RazorVarProducer",
    inputTag = cms.InputTag("scoutingRHemisphere"),
    inputMetTag = cms.InputTag("hltMetClean"),
)

scoutingRazorVarAnalyzer = cms.EDAnalyzer("RazorVarAnalyzer",
  modulePath=cms.untracked.string("Razor"),
  razorVarCollectionName=cms.untracked.InputTag("scoutingRazorVariables")
  )


#this file contains the sequence for data scouting using the Razor analysis
scoutingRazorDQMSequence = cms.Sequence(cms.ignore(scoutingRHemisphere)*
                                        scoutingRazorVariables*
                                        scoutingRazorVarAnalyzer
                                        )
