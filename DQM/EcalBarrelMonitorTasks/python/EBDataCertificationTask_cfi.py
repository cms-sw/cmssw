import FWCore.ParameterSet.Config as cms

ecalBarrelDataCertificationTask = cms.EDAnalyzer("EBDataCertificationTask",
    cloneME = cms.untracked.bool(False),
    prefixME = cms.untracked.string('EcalBarrel'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False)
)

# from DQM.EcalCommon.dqmpset import *
# from DQM.EcalCommon.CommonParams_cfi import *

# from DQM.EcalCommon.EcalDQMBinningService_cfi import *

# import DQM.EcalBarrelMonitorClient.CertificationClient_cfi as ecalCertificationClient

# from DQM.EcalBarrelMonitorTasks.EcalMonitorTask_cfi import ecalMonitorTaskPaths

# ecalMonitorClientPaths = dict(
#     CertificationClient = ecalCertificationClient.certificationClientPaths
# )

# ecalMonitorClientSources = dict(ecalMonitorClientPaths)
# ecalMonitorClientSources.update(ecalMonitorTaskPaths)

# ecalMonitorClientParams = dict(
#     CertificationClient = ecalCertificationClient.certificationClient,
#     Common = ecalCommonParams,
#     sources = dqmpaths("Ecal", ecalMonitorClientSources)
# )

# ecalBarrelDataCertificationTask = cms.EDAnalyzer("EcalDQMonitorClient",
#     moduleName = cms.untracked.string("Ecal Certification Summary"),
#     # tasks to be turned on
#     clients = cms.untracked.vstring(
#         "CertificationClient"
#     ),
#     # task parameters (included from indivitual cfis)
#     clientParameters = dqmpset(ecalMonitorClientParams),
#     # ME paths for each task (included from inidividual cfis)
#     mePaths = dqmpaths("Ecal", ecalMonitorClientPaths),
#     runAtEndLumi = cms.untracked.bool(True),
#     verbosity = cms.untracked.int32(0)
# )

