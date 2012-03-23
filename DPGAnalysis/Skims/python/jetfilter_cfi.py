import FWCore.ParameterSet.Config as cms

jetfilter = cms.EDFilter("SimpleJetFilter",
                         jetCollection = cms.InputTag("ak5CaloJets"),
                         jetIDMap = cms.InputTag("ak5JetID"),
                         ptCut = cms.double(30.),
                         maxRapidityCut = cms.double(999.0),
                         nJetMin = cms.uint32(2)
                         )

