import FWCore.ParameterSet.Config as cms


################# Photon Certification #########################
photonDataCertification = cms.EDProducer("PhotonDataCertification",
                              verbose = cms.bool(False)
                                         )
