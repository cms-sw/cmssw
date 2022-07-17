import FWCore.ParameterSet.Config as cms

particleFlowCandidateChecker = cms.EDAnalyzer(
    "PFCandidateChecker",
    pfCandidatesReco = cms.InputTag("particleFlow","","RECO"),
    pfCandidatesReReco = cms.InputTag("particleFlow","","REPROD"),
    pfJetsReco = cms.InputTag("ak5PFJets","","RECO"),
    pfJetsReReco = cms.InputTag("ak5PFJets","","REPROD"),
    deltaEMax = cms.double(1E-5),
    deltaEtaMax = cms.double(1E-5),
    deltaPhiMax = cms.double(1E-5),
    printBlocks = cms.untracked.bool(False),
    rankByPt = cms.untracked.bool(True)
    )
