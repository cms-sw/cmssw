import FWCore.ParameterSet.Config as cms

hiGenParticles = cms.EDProducer('GenParticleProducer',
                                mix = cms.string("mix"),
                                doSubEvent = cms.untracked.bool(True),
                                useCrossingFrame = cms.untracked.bool(True),
                                saveBarCodes = cms.untracked.bool(True),
                                abortOnUnknownPDGCode = cms.untracked.bool(False)
                               )
