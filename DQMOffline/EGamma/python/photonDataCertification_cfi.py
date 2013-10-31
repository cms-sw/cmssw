import FWCore.ParameterSet.Config as cms


################# Photon Certification #########################
photonDataCertification = cms.EDAnalyzer("PhotonDataCertification",
                              verbose = cms.bool(False)
                                         )
