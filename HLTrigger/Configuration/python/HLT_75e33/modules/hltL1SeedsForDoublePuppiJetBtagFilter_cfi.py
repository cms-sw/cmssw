import FWCore.ParameterSet.Config as cms

hltL1SeedsForDoublePuppiJetBtagFilter = cms.EDFilter("HLTP2GTSingleObjectFilter",
                                           saveTags = cms.bool(True),
                                           l1GTAlgoBlockTag = cms.InputTag('l1tGTAlgoBlockProducer'),
                                           minN = cms.uint32(2),
                                           l1GTAlgos = cms.VPSet(
                                               cms.PSet(
                                                   name = cms.string('pDoublePuppiJet112_112'),
                                                   collection = cms.PSet(
                                                       objectType = cms.string('CL2JetsSC4'),
                                                       minPt = cms.double(0),
                                                       maxAbsEta = cms.double(9999.)
                                                   )
                                               )
                                           ))
