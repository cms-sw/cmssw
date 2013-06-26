import FWCore.ParameterSet.Config as cms

#########################################################
## 95%
selection_95relIso = cms.PSet (
    trackIso_EB = cms.untracked.double(1.5e-01),
    ecalIso_EB =  cms.untracked.double(2.0e+00),
    hcalIso_EB =  cms.untracked.double(1.2e-01),
    sihih_EB =    cms.untracked.double(1.0e-02),
    dphi_EB =     cms.untracked.double(8.0e-01),
    deta_EB =     cms.untracked.double(7.0e-03),
    hoe_EB =      cms.untracked.double(1.5e-01),
    cIso_EB =     cms.untracked.double(10000.),
    
    trackIso_EE = cms.untracked.double(8.0e-02),
    ecalIso_EE =  cms.untracked.double(6.0e-02),
    hcalIso_EE =  cms.untracked.double(5.0e-02),
    sihih_EE =    cms.untracked.double(3.0e-02),
    dphi_EE =     cms.untracked.double(7.0e-01),
    deta_EE =     cms.untracked.double(1.0e-02),
    hoe_EE =      cms.untracked.double(7.0e-02),
    cIso_EE =     cms.untracked.double(10000. ),
    useConversionRejection = cms.untracked.bool(False),
    useExpectedMissingHits = cms.untracked.bool(True),
    maxNumberOfExpectedMissingHits = cms.untracked.int32(1),
    )

selection_95cIso = cms.PSet (
    trackIso_EB = cms.untracked.double(100000.),
    ecalIso_EB =  cms.untracked.double(100000.),
    hcalIso_EB =  cms.untracked.double(100000.),
    sihih_EB =    cms.untracked.double(1.0e-02),
    dphi_EB =     cms.untracked.double(8.0e-01),
    deta_EB =     cms.untracked.double(7.0e-03),
    hoe_EB =      cms.untracked.double(1.5e-01),
    cIso_EB =     cms.untracked.double(1.5e-01),
    
    trackIso_EE = cms.untracked.double(100000.),
    ecalIso_EE =  cms.untracked.double(100000.),
    hcalIso_EE =  cms.untracked.double(100000.),
    sihih_EE =    cms.untracked.double(3.0e-02),
    dphi_EE =     cms.untracked.double(7.0e-01),
    deta_EE =     cms.untracked.double(1.0e-02),
    hoe_EE =      cms.untracked.double(7.0e-02),
    cIso_EE =     cms.untracked.double(1.0e-01),
    useConversionRejection = cms.untracked.bool(False),
    useExpectedMissingHits = cms.untracked.bool(True),
    maxNumberOfExpectedMissingHits = cms.untracked.int32(1),
    )

#########################################################
##  90%
selection_90relIso = cms.PSet (
    trackIso_EB = cms.untracked.double(1.2e-01),
    ecalIso_EB =  cms.untracked.double(9.0e-02),
    hcalIso_EB =  cms.untracked.double(1.0e-01),
    sihih_EB =    cms.untracked.double(1.0e-02),
    dphi_EB =     cms.untracked.double(8.0e-01),
    deta_EB =     cms.untracked.double(7.0e-03),
    hoe_EB =      cms.untracked.double(1.2e-01),
    cIso_EB =     cms.untracked.double(10000. ),
    
    trackIso_EE = cms.untracked.double(5.0e-02),
    ecalIso_EE =  cms.untracked.double(6.0e-02),
    hcalIso_EE =  cms.untracked.double(3.0e-02),
    sihih_EE =    cms.untracked.double(3.0e-02),
    dphi_EE =     cms.untracked.double(7.0e-01),
    deta_EE =     cms.untracked.double(9.0e-03),
    hoe_EE =      cms.untracked.double(5.0e-02),
    cIso_EE =     cms.untracked.double(10000. ),
    useConversionRejection = cms.untracked.bool(True),
    useExpectedMissingHits = cms.untracked.bool(True),
    maxNumberOfExpectedMissingHits = cms.untracked.int32(1),
    )

selection_90cIso = cms.PSet (
    trackIso_EB = cms.untracked.double(100000.),
    ecalIso_EB =  cms.untracked.double(100000.),
    hcalIso_EB =  cms.untracked.double(100000.),
    sihih_EB =    cms.untracked.double(1.0e-02),
    dphi_EB =     cms.untracked.double(8.0e-01),
    deta_EB =     cms.untracked.double(7.0e-03),
    hoe_EB =      cms.untracked.double(1.2e-01),
    cIso_EB =     cms.untracked.double(1.0e-01),
    
    trackIso_EE = cms.untracked.double(100000.),
    ecalIso_EE =  cms.untracked.double(100000.),
    hcalIso_EE =  cms.untracked.double(100000.),
    sihih_EE =    cms.untracked.double(3.0e-02),
    dphi_EE =     cms.untracked.double(7.0e-01),
    deta_EE =     cms.untracked.double(9.0e-03),
    hoe_EE =      cms.untracked.double(5.0e-02),
    cIso_EE =     cms.untracked.double(7.0e-02),
    useConversionRejection = cms.untracked.bool(True),
    useExpectedMissingHits = cms.untracked.bool(True),
    maxNumberOfExpectedMissingHits = cms.untracked.int32(1),
    )

#########################################################
##  85%
selection_85relIso = cms.PSet (
    trackIso_EB = cms.untracked.double(9.0e-02),
    ecalIso_EB =  cms.untracked.double(8.0e-02),
    hcalIso_EB =  cms.untracked.double(1.0e-01),
    sihih_EB =    cms.untracked.double(1.0e-02),
    dphi_EB =     cms.untracked.double(6.0e-02),
    deta_EB =     cms.untracked.double(6.0e-03),
    hoe_EB =      cms.untracked.double(4.0e-02),
    cIso_EB =     cms.untracked.double(10000. ),
    
    trackIso_EE = cms.untracked.double(5.0e-02),
    ecalIso_EE =  cms.untracked.double(5.0e-02),
    hcalIso_EE =  cms.untracked.double(2.5e-02),
    sihih_EE =    cms.untracked.double(3.0e-02),
    dphi_EE =     cms.untracked.double(4.0e-02),
    deta_EE =     cms.untracked.double(7.0e-03),
    hoe_EE =      cms.untracked.double(2.5e-02),
    cIso_EE =     cms.untracked.double(10000. ),
    useConversionRejection = cms.untracked.bool(True),
    useExpectedMissingHits = cms.untracked.bool(True),
    maxNumberOfExpectedMissingHits = cms.untracked.int32(1),
    )

selection_85cIso = cms.PSet (
    trackIso_EB = cms.untracked.double(100000.),
    ecalIso_EB =  cms.untracked.double(100000.),
    hcalIso_EB =  cms.untracked.double(100000.),
    sihih_EB =    cms.untracked.double(1.0e-02),
    dphi_EB =     cms.untracked.double(6.0e-02),
    deta_EB =     cms.untracked.double(6.0e-03),
    hoe_EB =      cms.untracked.double(4.0e-02),    
    cIso_EB =     cms.untracked.double(9.0e-02),
    
    trackIso_EE = cms.untracked.double(100000.),
    ecalIso_EE =  cms.untracked.double(100000.),
    hcalIso_EE =  cms.untracked.double(100000.),
    sihih_EE =    cms.untracked.double(3.0e-02),
    dphi_EE =     cms.untracked.double(4.0e-02),
    deta_EE =     cms.untracked.double(7.0e-03),
    hoe_EE =      cms.untracked.double(2.5e-02),
    cIso_EE =     cms.untracked.double(6.0e-02),
    useConversionRejection = cms.untracked.bool(True),
    useExpectedMissingHits = cms.untracked.bool(True),
    maxNumberOfExpectedMissingHits = cms.untracked.int32(1),
    )

#########################################################
##  80%
selection_80relIso = cms.PSet (
    trackIso_EB = cms.untracked.double(9.0e-02),
    ecalIso_EB =  cms.untracked.double(7.0e-02),
    hcalIso_EB =  cms.untracked.double(1.0e-01),
    sihih_EB =    cms.untracked.double(1.0e-02),
    dphi_EB =     cms.untracked.double(6.0e-02),
    deta_EB =     cms.untracked.double(4.0e-03),
    hoe_EB =      cms.untracked.double(4.0e-02),
    cIso_EB =     cms.untracked.double(100000.),
    
    trackIso_EE = cms.untracked.double(4.0e-02),
    ecalIso_EE =  cms.untracked.double(5.0e-02),
    hcalIso_EE =  cms.untracked.double(2.5e-02),
    sihih_EE =    cms.untracked.double(3.0e-02),
    dphi_EE =     cms.untracked.double(3.0e-02),
    deta_EE =     cms.untracked.double(7.0e-03),
    hoe_EE =      cms.untracked.double(2.5e-02),
    cIso_EE =     cms.untracked.double(100000.),
    useConversionRejection = cms.untracked.bool(True),
    useExpectedMissingHits = cms.untracked.bool(True),
    maxNumberOfExpectedMissingHits = cms.untracked.int32(0),
    )

selection_80cIso = cms.PSet (
    trackIso_EB = cms.untracked.double(100000.),
    ecalIso_EB =  cms.untracked.double(100000.),
    hcalIso_EB =  cms.untracked.double(100000.),
    sihih_EB =    cms.untracked.double(1.0e-02),
    dphi_EB =     cms.untracked.double(6.0e-02),
    deta_EB =     cms.untracked.double(4.0e-03),
    hoe_EB =      cms.untracked.double(4.0e-02),
    cIso_EB =     cms.untracked.double(7.0e-02),
    
    trackIso_EE = cms.untracked.double(100000.),
    ecalIso_EE =  cms.untracked.double(100000.),
    hcalIso_EE =  cms.untracked.double(100000.),
    sihih_EE =    cms.untracked.double(3.0e-02),
    dphi_EE =     cms.untracked.double(3.0e-02),
    deta_EE =     cms.untracked.double(7.0e-03),
    hoe_EE =      cms.untracked.double(2.5e-02),
    cIso_EE =     cms.untracked.double(6.0e-02),
    useConversionRejection = cms.untracked.bool(True),
    useExpectedMissingHits = cms.untracked.bool(True),
    maxNumberOfExpectedMissingHits = cms.untracked.int32(0),
    )

#########################################################
##  70% corrected with lower limits to cut values
selection_70relIso = cms.PSet (
    trackIso_EB = cms.untracked.double(5.0e-02),
    ecalIso_EB =  cms.untracked.double(6.0e-02),
    hcalIso_EB =  cms.untracked.double(3.0e-02),
    sihih_EB =    cms.untracked.double(1.0e-02),
    dphi_EB =     cms.untracked.double(3.0e-02),
    deta_EB =     cms.untracked.double(4.0e-03),
    hoe_EB =      cms.untracked.double(2.5e-02),
    cIso_EB =     cms.untracked.double(100000.),
    
    trackIso_EE = cms.untracked.double(2.5e-02),
    ecalIso_EE =  cms.untracked.double(2.5e-02),
    hcalIso_EE =  cms.untracked.double(2.0e-02),
    sihih_EE =    cms.untracked.double(3.0e-02),
    dphi_EE =     cms.untracked.double(2.0e-02),
    deta_EE =     cms.untracked.double(5.0e-03),
    hoe_EE =      cms.untracked.double(2.5e-02),
    cIso_EE =     cms.untracked.double(100000.),
    useConversionRejection = cms.untracked.bool(True),
    useExpectedMissingHits = cms.untracked.bool(True),
    maxNumberOfExpectedMissingHits = cms.untracked.int32(0),
    )

selection_70cIso = cms.PSet (
    trackIso_EB = cms.untracked.double(100000.),
    ecalIso_EB =  cms.untracked.double(100000.),
    hcalIso_EB =  cms.untracked.double(100000.),
    sihih_EB =    cms.untracked.double(1.0e-02),
    dphi_EB =     cms.untracked.double(3.0e-02),
    deta_EB =     cms.untracked.double(4.0e-03),
    hoe_EB =      cms.untracked.double(2.5e-02),
    cIso_EB =     cms.untracked.double(4.0e-02),
    
    trackIso_EE = cms.untracked.double(100000.),
    ecalIso_EE =  cms.untracked.double(100000.),
    hcalIso_EE =  cms.untracked.double(100000.),
    sihih_EE =    cms.untracked.double(3.0e-02),
    dphi_EE =     cms.untracked.double(2.0e-02),
    deta_EE =     cms.untracked.double(5.0e-03),
    hoe_EE =      cms.untracked.double(2.5e-02),
    cIso_EE =     cms.untracked.double(3.0e-02),
    useConversionRejection = cms.untracked.bool(True),
    useExpectedMissingHits = cms.untracked.bool(True),
    maxNumberOfExpectedMissingHits = cms.untracked.int32(0),
    )

#########################################################
##  60% corrected with lower limits to cut values
selection_60relIso = cms.PSet (
    trackIso_EB = cms.untracked.double(4.0e-02),
    ecalIso_EB =  cms.untracked.double(4.0e-02),
    hcalIso_EB =  cms.untracked.double(3.0e-02),
    sihih_EB =    cms.untracked.double(1.0e-02),
    dphi_EB =     cms.untracked.double(2.5e-02),
    deta_EB =     cms.untracked.double(4.0e-03),
    hoe_EB =      cms.untracked.double(2.5e-02),
    cIso_EB =     cms.untracked.double(100000.),
    
    trackIso_EE = cms.untracked.double(2.5e-02),
    ecalIso_EE =  cms.untracked.double(2.0e-02),
    hcalIso_EE =  cms.untracked.double(2.0e-02),
    sihih_EE =    cms.untracked.double(3.0e-02),
    dphi_EE =     cms.untracked.double(2.0e-02),
    deta_EE =     cms.untracked.double(5.0e-03),
    hoe_EE =      cms.untracked.double(2.5e-02),
    cIso_EE =     cms.untracked.double(100000.),
    useConversionRejection = cms.untracked.bool(True),
    useExpectedMissingHits = cms.untracked.bool(True),
    maxNumberOfExpectedMissingHits = cms.untracked.int32(0),
    )

selection_60cIso = cms.PSet (
    trackIso_EB = cms.untracked.double(100000.),
    ecalIso_EB =  cms.untracked.double(100000.),
    hcalIso_EB =  cms.untracked.double(100000.),
    sihih_EB =    cms.untracked.double(1.0e-02),
    dphi_EB =     cms.untracked.double(2.5e-02),
    deta_EB =     cms.untracked.double(4.0e-03),
    hoe_EB =      cms.untracked.double(2.5e-02),
    cIso_EB =     cms.untracked.double(3.0e-02),
    
    trackIso_EE = cms.untracked.double(100000.),
    ecalIso_EE =  cms.untracked.double(100000.),
    hcalIso_EE =  cms.untracked.double(100000.),
    sihih_EE =    cms.untracked.double(3.0e-02),
    dphi_EE =     cms.untracked.double(2.0e-02),
    deta_EE =     cms.untracked.double(5.0e-03),
    hoe_EE =      cms.untracked.double(2.5e-02),
    cIso_EE =     cms.untracked.double(2.0e-02),
    useConversionRejection = cms.untracked.bool(True),
    useExpectedMissingHits = cms.untracked.bool(True),
    maxNumberOfExpectedMissingHits = cms.untracked.int32(0),
    )

#########################################################



