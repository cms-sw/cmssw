import FWCore.ParameterSet.Config as cms

hltEGL1SeedsForSinglePhotonIsolatedFilter = cms.EDFilter("HLTP2GTSingleObjectFilter",
                                                         saveTags = cms.bool(True),
                                                         l1GTAlgoBlockTag = cms.InputTag('l1tGTAlgoBlockProducer'),
                                                         minN = cms.uint32(1),
                                                         l1GTAlgos = cms.VPSet(
                                                             cms.PSet(
                                                                 name = cms.string('pSingleEGEle51'),
                                                                 collection = cms.PSet(
                                                                     objectType = cms.string('CL2Photons'),
                                                                     minPt = cms.double(0),
                                                                     maxAbsEta = cms.double(99999.)
                                                                 )
                                                             ),
                                                             cms.PSet(
                                                                 name = cms.string('pSingleIsoTkPho36'),
                                                                 collection = cms.PSet(
                                                                     objectType = cms.string('CL2Photons'),
                                                                     minPt = cms.double(0),
                                                                     maxAbsEta = cms.double(99999.)
                                                                 )
                                                             )
                                                         ))
