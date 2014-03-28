import FWCore.ParameterSet.Config as cms

particleFlowClusterECALWithTimeSelected = cms.EDProducer(
    "PFClusterSelector",
    src = cms.InputTag('particleFlowClusterECALWithTimeUncorrected'),
    energyRanges = cms.vdouble(1., 2., 5., 20.),
    ## pad the timing cuts on the high side (with repeats) to avoid overflows
    timingCutsLowBarrel = cms.vdouble(-12., -6., -4., -4., -4., -4.),
    timingCutsHighBarrel = cms.vdouble(12., 6., 4., 4., 4., 4.),
    timingCutsLowEndcap = cms.vdouble(-31.5, -20.5, -12., -5., -5., -5.),
    timingCutsHighEndcap = cms.vdouble(31.5, 20.5, 12., 5., 5., 5.)
    )
    
