import FWCore.ParameterSet.Config as cms

hltHcalLaserFilter = cms.EDFilter( "HLTHcalLaserFilter",
                                   hcalDigiCollection = cms.InputTag("hltHcalDigis"),
                                   timeSlices = cms.vint32([3,4,5,6]),
                                   thresholdsfC = cms.vdouble([15.]),
                                   CalibCountFilterValues=cms.vint32([100]),
                                   CalibChargeFilterValues=cms.vdouble([-1]),
                                   maxTotalCalibCharge = cms.double( -1 ),
                                   maxAllowedHFcalib = cms.int32(10)
                                   )

hltHcalLaserFilterHBHEOnly = cms.EDFilter( "HLTHcalLaserFilter",
                                           hcalDigiCollection = cms.InputTag("hltHcalDigis"),
                                           timeSlices = cms.vint32([3,4,5,6]),
                                           thresholdsfC = cms.vdouble([15.]),
                                           CalibCountFilterValues=cms.vint32([-1]),
                                           CalibChargeFilterValues=cms.vdouble([-1]),
                                           maxTotalCalibCharge = cms.double( -1 ),
                                           maxAllowedHFcalib = cms.int32(-1)
                                           )

hltHcalLaserFilterHFOnly = cms.EDFilter( "HLTHcalLaserFilter",
                                         hcalDigiCollection = cms.InputTag("hltHcalDigis"),
                                         timeSlices = cms.vint32([3,4,5,6]),
                                         thresholdsfC = cms.vdouble([15.]),
                                         CalibCountFilterValues=cms.vint32([-1]),
                                         CalibChargeFilterValues=cms.vdouble([-1]),
                                         maxTotalCalibCharge = cms.double( -1 ),
                                         maxAllowedHFcalib = cms.int32(10)
                                   )
