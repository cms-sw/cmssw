import FWCore.ParameterSet.Config as cms

hiFJRhoProducer = cms.EDProducer('HiFJRhoProducer',
                                 jetSource = cms.InputTag('kt4PFJets'),
                                 nExcl = cms.untracked.uint32(2),
                                 etaMaxExcl = cms.untracked.double(2.),
                                 ptMinExcl = cms.untracked.double(20.),
                                 nExcl2 = cms.untracked.uint32(1),
                                 etaMaxExcl2 = cms.untracked.double(3.),
                                 ptMinExcl2 = cms.untracked.double(20.),
     					         etaRanges = cms.untracked.vdouble(-5., -3., -2., -1.5, -1., 1., 1.5, 2., 3., 5.)
                                 # etaRanges = cms.untracked.vdouble(-5., -3., -2.1, -1.3, 1.3, 2.1, 3., 5.)
)
