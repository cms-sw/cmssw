import FWCore.ParameterSet.Config as cms

hiFJRhoProducer = cms.EDProducer('HiFJRhoProducer',
                                 jetSource = cms.InputTag('kt4PFJets'),
                                 nExcl = cms.untracked.uint32(2)
)

