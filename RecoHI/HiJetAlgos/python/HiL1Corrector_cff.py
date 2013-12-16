import FWCore.ParameterSet.Config as cms


heavyIonL1SubtractedJets = cms.EDProducer('HiL1Subtractor',
                                          src    = cms.InputTag('ak4HiRecoJets'),
                                          jetType    = cms.string('CaloJet'),
                                          rhoTag    = cms.string('kt4PFJets')
                                          #createNewCollection = cms.untracked.bool(True),
                                          #fillDummyEntries = cms.untracked.bool(True)
                                          )

heavyIonL1Subtracted = cms.Sequence(heavyIonL1SubtractedJets)
