import FWCore.ParameterSet.Config as cms

#
# This object is used to make changes for different running scenarios
#

SiPixelPhase1Summary_Online = cms.EDAnalyzer("SiPixelPhase1Summary",
    TopFolderName = cms.string('PixelPhase1/Phase1_MechanicalView/'),
    RunOnEndLumi = cms.bool(True),
    RunOnEndJob = cms.bool(True)         
)

SiPixelPhase1Summary_Offline = cms.EDAnalyzer("SiPixelPhase1Summary",
    TopFolderName = cms.string('PixelPhase1/Phase1_MechanicalView/'),
    RunOnEndLumi = cms.bool(False),
    RunOnEndJob = cms.bool(True)         
)

ADCQTester = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/SiPixelPhase1Config/test/qTests/mean_adc_qualitytest_config.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(True),
    qtestOnEndJob = cms.untracked.bool(True),
    reportThreshold = cms.untracked.string("more")
)

ADCQTester_offline = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/SiPixelPhase1Config/test/qTests/mean_adc_qualitytest_config.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(False),
    qtestOnEndJob = cms.untracked.bool(True),
    reportThreshold = cms.untracked.string("more")
)

NumClustersQTester = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/SiPixelPhase1Config/test/qTests/mean_num_clusters_qualitytest_config.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(True),
    qtestOnEndJob = cms.untracked.bool(True),
    reportThreshold = cms.untracked.string("more")
)

NumClustersQTester_offline = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/SiPixelPhase1Config/test/qTests/mean_num_clusters_qualitytest_config.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(False),
    qtestOnEndJob = cms.untracked.bool(True),
    reportThreshold = cms.untracked.string("more")
)

NumDigisQTester = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/SiPixelPhase1Config/test/qTests/mean_num_digis_qualitytest_config.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(True),
    qtestOnEndJob = cms.untracked.bool(True),
    reportThreshold = cms.untracked.string("more")
)

NumDigisQTester_offline = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/SiPixelPhase1Config/test/qTests/mean_num_digis_qualitytest_config.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(False),
    qtestOnEndJob = cms.untracked.bool(True),
    reportThreshold = cms.untracked.string("more")
)

SizeQTester = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/SiPixelPhase1Config/test/qTests/mean_size_qualitytest_config.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(True),
    qtestOnEndJob = cms.untracked.bool(True),
    reportThreshold = cms.untracked.string("more")
)

SizeQTester_offline = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/SiPixelPhase1Config/test/qTests/mean_size_qualitytest_config.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(False),
    qtestOnEndJob = cms.untracked.bool(True),
    reportThreshold = cms.untracked.string("more")
)

ChargeQTester = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/SiPixelPhase1Config/test/qTests/mean_charge_qualitytest_config.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(True),
    qtestOnEndJob = cms.untracked.bool(True),
    reportThreshold = cms.untracked.string("more")
)

ChargeQTester_offline = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/SiPixelPhase1Config/test/qTests/mean_charge_qualitytest_config.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(False),
    qtestOnEndJob = cms.untracked.bool(True),
    reportThreshold = cms.untracked.string("more")
)

RunQTests_online = cms.Sequence(ADCQTester * NumClustersQTester * NumDigisQTester * SizeQTester * ChargeQTester)
RunQTests_offline = cms.Sequence(ADCQTester_offline * NumClustersQTester_offline * NumDigisQTester_offline * SizeQTester_offline * ChargeQTester_offline)
