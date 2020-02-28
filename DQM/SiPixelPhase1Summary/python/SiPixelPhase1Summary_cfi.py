import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

#
# This object is used to make changes for different running scenarios
#

SiPixelPhase1SummaryOnline = DQMEDHarvester("SiPixelPhase1Summary",
    TopFolderName = cms.string('PixelPhase1/Phase1_MechanicalView/'),
    RunOnEndLumi = cms.bool(True),
    RunOnEndJob = cms.bool(True),
    SummaryMaps = cms.VPSet(
        cms.PSet(
            MapName = cms.string("Digi"),
            MapHist = cms.string("mean_num_digis")
            ),
        cms.PSet(
            MapName = cms.string("ADC"),
            MapHist = cms.string("mean_adc")
            ),
        cms.PSet(
            MapName = cms.string("NClustsTotal"),
            MapHist = cms.string("mean_num_clusters")
            ),
        cms.PSet(
            MapName = cms.string("ClustWidthOnTrk"),
            MapHist = cms.string("mean_size")
            ),
        cms.PSet(
            MapName = cms.string("Charge"),
            MapHist = cms.string("mean_charge")
            )
        ),
    # Number of dead ROCs required to generate an error. Order must be layers 1-4, ring1, ring2.
    DeadROCErrorThreshold = cms.vdouble(0.2,0.2,0.2,0.2,0.2,0.2)
)

SiPixelPhase1SummaryOffline = DQMEDHarvester("SiPixelPhase1Summary",
    TopFolderName = cms.string('PixelPhase1/Phase1_MechanicalView/'),
    RunOnEndLumi = cms.bool(False),
    RunOnEndJob = cms.bool(True),
    SummaryMaps = cms.VPSet(
        cms.PSet(
            MapName = cms.string("Digi"),
            MapHist = cms.string("mean_num_digis")
            ),
        cms.PSet(
            MapName = cms.string("ADC"),
            MapHist = cms.string("mean_adc")
            ),
        cms.PSet(
            MapName = cms.string("NClustsTotal"),
            MapHist = cms.string("mean_num_clusters")
            ),
        cms.PSet(
            MapName = cms.string("ClustWidthOnTrk"),
            MapHist = cms.string("mean_size")
            ),
        cms.PSet(
            MapName = cms.string("Charge"),
            MapHist = cms.string("mean_charge")
            )
        ),
        DeadROCErrorThreshold = cms.vdouble(0.2,0.2,0.2,0.2,0.2,0.2)

)

SiPixelPhase1SummaryCosmics = DQMEDHarvester("SiPixelPhase1Summary",
    TopFolderName = cms.string('PixelPhase1/Phase1_MechanicalView/'),
    RunOnEndLumi = cms.bool(False),
    RunOnEndJob = cms.bool(True),
    SummaryMaps = cms.VPSet(
        cms.PSet(
            MapName = cms.string("Digi"),
            MapHist = cms.string("mean_num_digis")
            ),
        cms.PSet(
            MapName = cms.string("ClustWidthOnTrk"),
            MapHist = cms.string("mean_size")
            ),
        cms.PSet(
            MapName = cms.string("Charge"),
            MapHist = cms.string("mean_charge")
            )
        ),
        DeadROCErrorThreshold = cms.vdouble(0.2,0.2,0.2,0.2,0.2,0.2)
)

from DQMServices.Core.DQMQualityTester import DQMQualityTester
ADCQTester = DQMQualityTester(
    qtList = cms.untracked.FileInPath('DQM/SiPixelPhase1Config/test/qTests/mean_adc_qualitytest_config.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(True),
    qtestOnEndJob = cms.untracked.bool(True),
    reportThreshold = cms.untracked.string("more")
)

ADCQTester_offline = DQMQualityTester(
    qtList = cms.untracked.FileInPath('DQM/SiPixelPhase1Config/test/qTests/mean_adc_qualitytest_config.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(False),
    qtestOnEndJob = cms.untracked.bool(True),
    reportThreshold = cms.untracked.string("more")
)

NumClustersQTester = DQMQualityTester(
    qtList = cms.untracked.FileInPath('DQM/SiPixelPhase1Config/test/qTests/mean_num_clusters_qualitytest_config.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(True),
    qtestOnEndJob = cms.untracked.bool(True),
    reportThreshold = cms.untracked.string("more")
)

NumClustersQTester_offline = DQMQualityTester(
    qtList = cms.untracked.FileInPath('DQM/SiPixelPhase1Config/test/qTests/mean_num_clusters_qualitytest_config.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(False),
    qtestOnEndJob = cms.untracked.bool(True),
    reportThreshold = cms.untracked.string("more")
)

NumDigisQTester = DQMQualityTester(
    qtList = cms.untracked.FileInPath('DQM/SiPixelPhase1Config/test/qTests/mean_num_digis_qualitytest_config.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(True),
    qtestOnEndJob = cms.untracked.bool(True),
    reportThreshold = cms.untracked.string("more")
)

NumDigisQTester_offline = DQMQualityTester(
    qtList = cms.untracked.FileInPath('DQM/SiPixelPhase1Config/test/qTests/mean_num_digis_qualitytest_config.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(False),
    qtestOnEndJob = cms.untracked.bool(True),
    reportThreshold = cms.untracked.string("more")
)

NumDigisQTester_cosmics = DQMQualityTester(
    qtList = cms.untracked.FileInPath('DQM/SiPixelPhase1Config/test/qTests/mean_num_digis_qualitytest_config_cosmics.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(False),
    qtestOnEndJob = cms.untracked.bool(True),
    reportThreshold = cms.untracked.string("more")
)

SizeQTester = DQMQualityTester(
    qtList = cms.untracked.FileInPath('DQM/SiPixelPhase1Config/test/qTests/mean_size_qualitytest_config.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(True),
    qtestOnEndJob = cms.untracked.bool(True),
    reportThreshold = cms.untracked.string("more")
)

SizeQTester_offline = DQMQualityTester(
    qtList = cms.untracked.FileInPath('DQM/SiPixelPhase1Config/test/qTests/mean_size_qualitytest_config.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(False),
    qtestOnEndJob = cms.untracked.bool(True),
    reportThreshold = cms.untracked.string("more")
)

SizeQTester_cosmics = DQMQualityTester(
    qtList = cms.untracked.FileInPath('DQM/SiPixelPhase1Config/test/qTests/mean_size_qualitytest_config_cosmics.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(False),
    qtestOnEndJob = cms.untracked.bool(True),
    reportThreshold = cms.untracked.string("more")
)

ChargeQTester = DQMQualityTester(
    qtList = cms.untracked.FileInPath('DQM/SiPixelPhase1Config/test/qTests/mean_charge_qualitytest_config.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(True),
    qtestOnEndJob = cms.untracked.bool(True),
    reportThreshold = cms.untracked.string("more")
)

ChargeQTester_offline = DQMQualityTester(
    qtList = cms.untracked.FileInPath('DQM/SiPixelPhase1Config/test/qTests/mean_charge_qualitytest_config.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(False),
    qtestOnEndJob = cms.untracked.bool(True),
    reportThreshold = cms.untracked.string("more")
)

ChargeQTester_cosmics = DQMQualityTester(
    qtList = cms.untracked.FileInPath('DQM/SiPixelPhase1Config/test/qTests/mean_charge_qualitytest_config_cosmics.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True),
    qtestOnEndLumi = cms.untracked.bool(False),
    qtestOnEndJob = cms.untracked.bool(True),
    reportThreshold = cms.untracked.string("more")
)

RunQTests_online = cms.Sequence(ADCQTester * NumClustersQTester * NumDigisQTester * SizeQTester * ChargeQTester)
RunQTests_offline = cms.Sequence(ADCQTester_offline * NumClustersQTester_offline * NumDigisQTester_offline * SizeQTester_offline * ChargeQTester_offline)
RunQTests_cosmics = cms.Sequence(NumDigisQTester_cosmics * SizeQTester_cosmics * ChargeQTester_cosmics)
