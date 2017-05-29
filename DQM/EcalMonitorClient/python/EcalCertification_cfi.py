import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

from DQM.EcalCommon.CommonParams_cfi import ecalCommonParams

from DQM.EcalMonitorClient.CertificationClient_cfi import ecalCertificationClient

ecalCertification = DQMEDHarvester("EcalDQMonitorClient",
    moduleName = cms.untracked.string("Ecal Certification Client"),
    # workers to be turned on
    workers = cms.untracked.vstring(
        "CertificationClient"
    ),
    # task parameters (included from indivitual cfis)
    workerParameters = cms.untracked.PSet(
        CertificationClient = ecalCertificationClient.clone()
    ),
    commonParameters = ecalCommonParams.clone(willConvertToEDM = cms.untracked.bool(False)),
    verbosity = cms.untracked.int32(0)
)
