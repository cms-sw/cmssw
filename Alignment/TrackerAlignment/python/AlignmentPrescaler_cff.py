import FWCore.ParameterSet.Config as cms

AlignmentPrescaler = cms.EDProducer("AlignmentPrescaler",
                                    src = cms.InputTag('generalTracks'),
                                    assomap=cms.InputTag('OverlapAssoMap'),
                                    PrescFileName=cms.string('PrescaleFactors.root'),
                                    PrescTreeName=cms.string('AlignmentHitMap')#if you change this be sure to be consistent with the rest of your code
                                    )
# foo bar baz
# y3DSuygbNwB0M
# h9ebcgwD60Zbq
