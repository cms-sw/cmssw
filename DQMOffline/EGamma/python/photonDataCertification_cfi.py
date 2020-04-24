import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester


################# Photon Certification #########################
photonDataCertification = DQMEDHarvester("PhotonDataCertification",
                              verbose = cms.bool(False)
                                         )
