import FWCore.ParameterSet.Config as cms

particleFlowClusterECALWithTimeSelected = cms.EDProducer(
    "PFClusterSelector",
    src = cms.InputTag('particleFlowClusterECALWithTimeUncorrected'),
    energyRanges = cms.vdouble(0.,0.5,1.,2.,5.,5.),
    timingCutsLow = cms.vdouble(-16.,-16.,-16.,-12.,-10.,-10.),
    timingCutsHigh = cms.vdouble(16.,16.,16.,12.,10.,10.),
    timingCutsEndcapLow = cms.vdouble(-18.,-18.,-18.,-18.,-14.,-14.),
    timingCutsEndcapHigh = cms.vdouble(18.,18.,18.,18.,14.,14.)
    )
