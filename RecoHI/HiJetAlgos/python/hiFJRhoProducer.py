import FWCore.ParameterSet.Config as cms

hiFJRhoProducer = cms.EDProducer('HiFJRhoProducer',
                                 jetSource = cms.InputTag('kt4PFJets'),
                                 nExcl = cms.untracked.uint32(2),
                                 etaMaxExcl = cms.untracked.double(2.),
                                 ptMinExcl = cms.untracked.double(20.),
                                 nExcl2 = cms.untracked.uint32(1),
                                 etaMaxExcl2 = cms.untracked.double(3.),
                                 ptMinExcl2 = cms.untracked.double(20.)
)

