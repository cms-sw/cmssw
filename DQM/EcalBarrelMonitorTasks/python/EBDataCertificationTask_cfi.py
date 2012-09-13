import FWCore.ParameterSet.Config as cms

from DQM.EcalCommon.dqmpset import *
from DQM.EcalCommon.CommonParams_cfi import *

from DQM.EcalBarrelMonitorClient.CertificationClient_cfi import ecalCertificationClient

from DQM.EcalCommon.EcalDQMBinningService_cfi import *

ecalBarrelDataCertificationTask = cms.EDAnalyzer("EcalDQMonitorClient",
    moduleName = cms.untracked.string("Ecal Certification Summary"),
    mergeRuns = cms.untracked.bool(False),
    # tasks to be turned on
    workers = cms.untracked.vstring(
        "CertificationClient"
    ),
    # task parameters (included from indivitual cfis)
    workerParameters = dqmpset(
        dict(
            CertificationClient = ecalCertificationClient,
            common = ecalCommonParams
        )
    ),
    verbosity = cms.untracked.int32(0)
)
