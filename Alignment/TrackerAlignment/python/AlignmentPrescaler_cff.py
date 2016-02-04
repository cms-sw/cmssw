import FWCore.ParameterSet.Config as cms

AlignmentPrescaler = cms.EDProducer("AlignmentPrescaler",
                                    src = cms.InputTag('generalTracks'),
                                    assomap=cms.InputTag('OverlapAssoMap'),
                                    PrescFileName=cms.string('PrescaleFactors.root'),
                                    PrescTreeName=cms.string('AlignmentHitMap')#if you change this be sure to be consistent with the rest of your code
                                    )
