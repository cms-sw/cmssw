import FWCore.ParameterSet.Config as cms

hcalDataCertification = cms.EDAnalyzer('HcalDataCertification',
                                       subSystemFolder = cms.untracked.string("Hcal")
)
