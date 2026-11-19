import FWCore.ParameterSet.Config as cms

hltEGL1SeedsForSingleEleIsolatedFilter = cms.EDFilter("HLTP2GTSingleObjectFilter",
                                                      saveTags = cms.bool(True),
                                                      l1GTAlgoBlockTag = cms.InputTag('l1tGTAlgoBlockProducer'),
                                                      minN = cms.uint32(1),
                                                      l1GTAlgos = cms.VPSet(
                                                          cms.PSet(
                                                              name = cms.string('pSingleEGEle51'),
                                                              collection = cms.PSet(
                                                                  objectType = cms.string('CL2Photons'),
                                                                  minPt = cms.double(0),
                                                                  maxAbsEta = cms.double(9999.)
                                                              )
                                                          ),
                                                          cms.PSet(
                                                              name = cms.string('pSingleTkEle36'),
                                                              collection = cms.PSet(
                                                                  objectType = cms.string('CL2Electrons'),
                                                                  minPt = cms.double(0),
                                                                  maxAbsEta = cms.double(9999.)
                                                              )
                                                          ),
                                                          cms.PSet(
                                                              name = cms.string('pSingleIsoTkEle28'),
                                                              collection = cms.PSet(
                                                                  objectType = cms.string('CL2Electrons'),
                                                                  minPt = cms.double(0),
                                                                  maxAbsEta = cms.double(9999.)
                                                              )
                                                          )
                                                      ))
