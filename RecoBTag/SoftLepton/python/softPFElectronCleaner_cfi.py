import FWCore.ParameterSet.Config as cms

#79% signal eff at 5% background acceptance
looseSoftPFElectronCleanerBarrelCuts = cms.PSet( 
    BarrelPtCuts                    = cms.untracked.vdouble(2.0, 9999.0),
    BarreldRGsfTrackElectronCuts    = cms.untracked.vdouble(0.0, 0.017),
    BarrelEemPinRatioCuts           = cms.untracked.vdouble(-0.9, 0.39),
    BarrelMVACuts                   = cms.untracked.vdouble(-0.1, 1.0),
    BarrelInversedRFirstLastHitCuts = cms.untracked.vdouble(0.0, 77.7),
    BarrelRadiusFirstHitCuts        = cms.untracked.vdouble(0.0, 10.0),
    BarrelZFirstHitCuts             = cms.untracked.vdouble(-50.0, 50.0)
)
#75% signal eff at 5% background acceptance
looseSoftPFElectronCleanerForwardCuts = cms.PSet(
    ForwardPtCuts                     = cms.untracked.vdouble(2.0, 9999.0),
    ForwardInverseFBremCuts           = cms.untracked.vdouble(1.0, 7.01),
    ForwarddRGsfTrackElectronCuts     = cms.untracked.vdouble(0.0, 0.006),
    ForwardRadiusFirstHitCuts         = cms.untracked.vdouble(0.0, 6.35),
    ForwardZFirstHitCuts              = cms.untracked.vdouble(-141.0, 141.0),
    ForwardMVACuts                    = cms.untracked.vdouble(-0.24, 1.0)
)

#49% signal eff at 1% background acceptance
mediumSoftPFElectronCleanerBarrelCuts = cms.PSet( 
    BarrelPtCuts                      = cms.untracked.vdouble(2.0, 9999.0),
    BarreldRGsfTrackElectronCuts      = cms.untracked.vdouble(0.0, 0.0047),
    BarrelEemPinRatioCuts             = cms.untracked.vdouble(-0.9, 0.54),
    BarrelMVACuts                     = cms.untracked.vdouble(0.6, 1.0),
    BarrelInversedRFirstLastHitCuts   = cms.untracked.vdouble(0.0, 80.0),
    BarrelRadiusFirstHitCuts          = cms.untracked.vdouble(0.0, 10.0),
    BarrelZFirstHitCuts               = cms.untracked.vdouble(-33.2, 33.2)
)
#48% signal eff at 1% background acceptance
mediumSoftPFElectronCleanerForwardCuts = cms.PSet(
    ForwardPtCuts                      = cms.untracked.vdouble(2.0, 9999.0),
    ForwardInverseFBremCuts            = cms.untracked.vdouble(1.0, 20.0),
    ForwarddRGsfTrackElectronCuts      = cms.untracked.vdouble(0.0, 0.003),
    ForwardRadiusFirstHitCuts          = cms.untracked.vdouble(0.0, 6.35),
    ForwardZFirstHitCuts               = cms.untracked.vdouble(-186.0, 186.0),
    ForwardMVACuts                     = cms.untracked.vdouble(0.37, 1.0)
)

#34% signal eff at 0.5% background acceptance
tightSoftPFElectronCleanerBarrelCuts = cms.PSet( 
    BarrelPtCuts                     = cms.untracked.vdouble(2.0, 9999.0),
    BarreldRGsfTrackElectronCuts     = cms.untracked.vdouble(0.0, 0.006),
    BarrelEemPinRatioCuts            = cms.untracked.vdouble(-0.9, 0.065),
    BarrelMVACuts                    = cms.untracked.vdouble(0.58, 1.0),
    BarrelInversedRFirstLastHitCuts  = cms.untracked.vdouble(0.0, 10.7),
    BarrelRadiusFirstHitCuts         = cms.untracked.vdouble(0.0, 10.0),
    BarrelZFirstHitCuts              = cms.untracked.vdouble(-8.0, 8.0)
)
#22% signal eff at 0.5% background acceptance
tightSoftPFElectronCleanerForwardCuts = cms.PSet(
    ForwardPtCuts                     = cms.untracked.vdouble(2.0, 9999.0),
    ForwardInverseFBremCuts           = cms.untracked.vdouble(1.0, 15.0),
    ForwarddRGsfTrackElectronCuts     = cms.untracked.vdouble(0.0, 0.01),
    ForwardRadiusFirstHitCuts         = cms.untracked.vdouble(0.0, 6.35),
    ForwardZFirstHitCuts              = cms.untracked.vdouble(-45.3, 45.3),
    ForwardMVACuts                    = cms.untracked.vdouble(0.60, 1.0)
)

