import FWCore.ParameterSet.Config as cms

hiFJRhoProducer = cms.EDProducer('HiFJRhoProducer',
                                 jetSource = cms.InputTag('kt4PFJetsForRho'),
                                 nExcl = cms.int32(2),
                                 etaMaxExcl = cms.double(2.),
                                 ptMinExcl = cms.double(20.),
                                 nExcl2 = cms.int32(1),
                                 etaMaxExcl2 = cms.double(3.),
                                 ptMinExcl2 = cms.double(20.),
     				 etaRanges = cms.vdouble(-5., -3., -2., -1.5, -1., 1., 1.5, 2., 3., 5.)
)
