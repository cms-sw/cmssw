import FWCore.ParameterSet.Config as cms

# JAN - currently doesn't make sense for mu-ele as no hadronic tau 
# corrections are needed in the mu-ele channel, but this should be replaced
# by something like an electron 

cmgMuEleCor = cms.EDProducer(
    "MuEleUpdateProducer",
    diObjectCollection  = cms.InputTag('cmgMuEle'),
    genCollection = cms.InputTag('prunedGenParticles'),
    nSigma              = cms.double(0),
    uncertainty         = cms.double(0.03), # 2012: 0.03
    shift1ProngNoPi0    = cms.double(0.),
    shift1Prong1Pi0     = cms.double(0.), # 2012: 0.012
    ptDependence1Pi0    = cms.double(0.),
    shift3Prong         = cms.double(0.), # 2012: 0.012
    ptDependence3Prong  = cms.double(0.),
    shiftMet = cms.bool(True),
    shiftTaus = cms.bool(True)
)