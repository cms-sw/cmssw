import FWCore.ParameterSet.Config as cms


################# Photon Certification #########################
photonDataCertification = cms.EDAnalyzer("PhotonDataCertification",
                              verbose = cms.bool(False)
                                         )


################# Photon Quality Tests  #########################
qTesterPhoton = cms.EDAnalyzer("QualityTester",
     qtList = cms.untracked.FileInPath('DQMOffline/EGamma/test/EGamma.xml'),
     prescaleFactor = cms.untracked.int32(1),
     testInEventloop = cms.untracked.bool(False),
     verboseQT =  cms.untracked.bool(False),
                         
 )
