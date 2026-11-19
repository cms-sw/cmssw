import FWCore.ParameterSet.Config as cms

hltPuppiTauTkIsoEle45and22L1TkFilter = cms.EDFilter("HLTP2GTDoubleObjectFilter",
                                                    l1GTAlgoBlockTag = cms.InputTag("l1tGTAlgoBlockProducer"),
                                                    l1GTAlgos = cms.VPSet(
                                                        cms.PSet(
                                                            name = cms.string("pPuppiTauTkIsoEle45_22"),
                                                            collection1 = cms.PSet(
                                                                objectType = cms.string("CL2Electrons"),
                                                                minPt      = cms.double(0.),
                                                                maxAbsEta  = cms.double(9999.),
                                                            ),
                                                            collection2 = cms.PSet(
                                                                objectType = cms.string("CL2Taus"),
                                                                minPt      = cms.double(0.),
                                                                maxAbsEta  = cms.double(9999.),
                                                            ),
                                                            minDR      = cms.double(0.0),
                                                            maxDR      = cms.double(1e9),
                                                            minDEta    = cms.double(-1.0),
                                                            minDPhi    = cms.double(-1.0),
                                                            minInvMass = cms.double(0.0),
                                                            maxInvMass = cms.double(1e9),
                                                        )
                                                    )
                                                  )
