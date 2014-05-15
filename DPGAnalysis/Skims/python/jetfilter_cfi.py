import FWCore.ParameterSet.Config as cms

jetfilter = cms.EDFilter("SimpleJetFilter",
                         jetCollection = cms.InputTag("ak4CaloJets"),
                         jetIDMap = cms.InputTag("ak4JetID"),
                         ptCut = cms.double(30.),
                         maxRapidityCut = cms.double(999.0),
                         nJetMin = cms.uint32(2)
                         )

