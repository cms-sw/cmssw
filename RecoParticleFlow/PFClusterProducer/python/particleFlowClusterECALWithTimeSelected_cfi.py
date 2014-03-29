import FWCore.ParameterSet.Config as cms

particleFlowClusterECALWithTimeSelected = cms.EDProducer(
    "PFClusterSelector",
    src = cms.InputTag('particleFlowClusterECALWithTimeUncorrected'),
    energyRanges = cms.vdouble(0.0,0.5,1.0,2.0,5.0),
    ## pad the timing cuts on the high side (with repeats) to aboid overflows
    timingCutsLowBarrel = cms.vdouble(-16.,-16.,-16.,-12.,-10.,-10.),
    timingCutsHighBarrel = cms.vdouble(16.,16.,16.,12.,10.,10.),
    timingCutsLowEndcap = cms.vdouble(-18.,-18.,-18.,-18.,-14.,-14.),
    timingCutsHighEndcap = cms.vdouble(18.,18.,18.,18.,14.,14.)
    )
