import FWCore.ParameterSet.Config as cms


hiSignalGenJets = cms.EDProducer('HiSignalGenJetProducer',
                                    src    = cms.InputTag('ak4HiGenJets')
                                    )

