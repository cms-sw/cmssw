import FWCore.ParameterSet.Config as cms

patElectronEAIso03CorrectionProducer = cms.EDProducer( "PatElectronEAIsoCorrectionProducer"
                                                     , patElectrons = cms.InputTag( 'patElectrons' )
                                                     , eaIsolator   = cms.InputTag( 'elPFIsoValueEA03' )
                                                     )
