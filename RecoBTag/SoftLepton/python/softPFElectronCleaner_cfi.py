import FWCore.ParameterSet.Config as cms

#79% signal eff at 5% background acceptance
looseSoftPFElectronCleanerBarrelCuts = cms.PSet( 
    BarrelPtCuts                    = cms.vdouble(2.0, 9999.0),
    BarreldRGsfTrackElectronCuts    = cms.vdouble(0.0, 0.017),
    BarrelEemPinRatioCuts           = cms.vdouble(-0.9, 0.39),
    BarrelMVACuts                   = cms.vdouble(-0.1, 1.0),
)
#75% signal eff at 5% background acceptance
looseSoftPFElectronCleanerForwardCuts = cms.PSet(
    ForwardPtCuts                     = cms.vdouble(2.0, 9999.0),
    ForwardInverseFBremCuts           = cms.vdouble(1.0, 7.01),
    ForwarddRGsfTrackElectronCuts     = cms.vdouble(0.0, 0.006),
    ForwardMVACuts                    = cms.vdouble(-0.24, 1.0)
)

#49% signal eff at 1% background acceptance
mediumSoftPFElectronCleanerBarrelCuts = cms.PSet( 
    BarrelPtCuts                      = cms.vdouble(2.0, 9999.0),
    BarreldRGsfTrackElectronCuts      = cms.vdouble(0.0, 0.0047),
    BarrelEemPinRatioCuts             = cms.vdouble(-0.9, 0.54),
    BarrelMVACuts                     = cms.vdouble(0.6, 1.0),
)
#48% signal eff at 1% background acceptance
mediumSoftPFElectronCleanerForwardCuts = cms.PSet(
    ForwardPtCuts                      = cms.vdouble(2.0, 9999.0),
    ForwardInverseFBremCuts            = cms.vdouble(1.0, 20.0),
    ForwarddRGsfTrackElectronCuts      = cms.vdouble(0.0, 0.003),
    ForwardMVACuts                     = cms.vdouble(0.37, 1.0)
)

#34% signal eff at 0.5% background acceptance
tightSoftPFElectronCleanerBarrelCuts = cms.PSet( 
    BarrelPtCuts                     = cms.vdouble(2.0, 9999.0),
    BarreldRGsfTrackElectronCuts     = cms.vdouble(0.0, 0.006),
    BarrelEemPinRatioCuts            = cms.vdouble(-0.9, 0.065),
    BarrelMVACuts                    = cms.vdouble(0.58, 1.0),
)
#22% signal eff at 0.5% background acceptance
tightSoftPFElectronCleanerForwardCuts = cms.PSet(
    ForwardPtCuts                     = cms.vdouble(2.0, 9999.0),
    ForwardInverseFBremCuts           = cms.vdouble(1.0, 15.0),
    ForwarddRGsfTrackElectronCuts     = cms.vdouble(0.0, 0.01),
    ForwardMVACuts                    = cms.vdouble(0.60, 1.0)
)

