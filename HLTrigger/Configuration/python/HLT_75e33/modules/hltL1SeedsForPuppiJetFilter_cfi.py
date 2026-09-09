import FWCore.ParameterSet.Config as cms

hltL1SeedsForPuppiJetFilter = cms.EDFilter("HLTP2GTSingleObjectFilter",
                                           saveTags = cms.bool(True),
                                           l1GTAlgoBlockTag = cms.InputTag('l1tGTAlgoBlockProducer'),
                                           minN = cms.uint32(1),
                                           l1GTAlgos = cms.VPSet(
                                               cms.PSet(
                                                   name = cms.string('pSinglePuppiJet230'),
                                                   collection = cms.PSet(
                                                       objectType = cms.string('CL2JetsSC4'),
                                                       minPt = cms.double(0),
                                                       maxAbsEta = cms.double(2.4)
                                                   )
                                               )
                                           ))
