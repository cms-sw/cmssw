import FWCore.ParameterSet.Config as cms

class config: pass
#config.runNumber = 186323
config.runNumber = 186760
#config.runNumber = 186744
#config.refTag = 'DT_t0_cosmic2009_V01_express'
config.refTag = 't0Fake_20X_Sept15_mc'
#config.connect_ref = 'oracle://cms_orcoff_prod/CMS_COND_31X_DT'
config.connect_ref = 'oracle://cms_orcoff_prod/CMS_COND_31X_FROM21X'

#config.t0DB = 'dtT0WireCalibration-Run186323-v4/res/t0_1_1_s1L.db'
#config.t0DB = 'dtT0WireCalibration-Run186760-v1/res/t0_1_1_gPo.db'
#config.t0DB = 'dtT0WireCalibration-Run186744-v2/res/t0_1_1_WKl.db'
#config.t0DB = 'Run186323-dtT0WireCalibration-Run186323-v4/Results/t0_correction_chamber_Wh2_MB2_Sec12_t0-MiniDaq-Run173264_186323.db'
#config.t0DB = 'Run186323-dtT0WireCalibration-Run186323-v4/Results/t0_correction_Ch_Wh2_MB2_Sec12-FillMissing-MiniDaq-Run173264_t0_186323.db'
#config.t0DB = 'Run186323-dtT0WireCalibration-Run186323-v4/Results/t0_correction_Ch_Wh2_MB2_Sec12-FillMissing-MiniDaq_Run2011A_Run173264_t0_186323.db'
#config.t0DB = 'Run186323-dtT0WireCalibration-Run186323-v4/Results/t0_correction_Ch_Wh-2_MB2_Sec12_Wh2_MB3_Sec12_Wh-1_MB4_Sec1-FillMissing-MiniDaq_Run2011A_Run173264_t0_186323.db'
#config.t0DB = 'dtT0WireCalibration-ReferenceWireInLayer-Run186323-v1/res/t0_1_1_YZD.db'
#config.t0DB = 'dtT0WireCalibration-ReferenceWireInLayer-Run186760-v1/res/t0_1_1_RA4.db'
#config.t0DB = 'Run186323-dtT0WireCalibration-AbsoluteT0-Run186323-v1/Results/t0_correction_absolute_reference_All_186323.db'
config.t0DB = 'Run186760-dtT0WireCalibration-AbsoluteT0-Run186760-v1/Results/t0_correction_absolute_reference_All_186760.db'

config.dataset = '/MiniDaq/Commissioning12-v1/RAW'
#config.desc = 'dtT0DBValidation'
#config.desc = 'dtT0DBValidation-NewThreshold'
#config.desc = 'dtT0DBValidation-T0Corr'
#config.desc = 'dtT0DBValidation-T0Corr_Ch_Wh2_MB2_Sec12'
#config.desc = 'dtT0DBValidation-T0Corr_Ch_Wh2_MB2_Sec12_FillMissing'
#config.desc = 'dtT0DBValidation-T0Corr_Ch_Wh2_MB2_Sec12_FillMissing_MiniDaq_Run2011A_Run173264'
#config.desc = 'dtT0DBValidation-T0Corr_Ch_Wh-2_MB2_Sec12_Wh2_MB3_Sec12_Wh-1_MB4_Sec1_FillMissing_MiniDaq_Run2011A_Run173264'
#config.desc = 'dtT0DBValidation_t0Fake_20X_Sept15_mc'
config.desc = 'dtT0DBValidation-AbsoluteReference_t0Fake_20X_Sept15_mc'
config.outputdir = '.'
config.trial = 1

# Further config.
dataset_vec = config.dataset.split('/')
config.workflowName = '/%s/%s-%s-rev%d/%s' % (dataset_vec[1],
                                              dataset_vec[2],
                                              config.desc,
                                              config.trial,
                                              dataset_vec[3])

from DQMOffline.CalibMuon.dtT0DBValidation_cfg import process
process.source.firstRun = config.runNumber
process.tzeroRef.toGet = cms.VPSet(
    cms.PSet(
        record = cms.string('DTT0Rcd'),
        tag = cms.string(config.refTag),
        connect = cms.untracked.string( config.connect_ref ),
        label = cms.untracked.string('tzeroRef')
    ),
    cms.PSet(
        record = cms.string('DTT0Rcd'),
        tag = cms.string('t0'),
        connect = cms.untracked.string('sqlite_file:%s' % config.t0DB),
        label = cms.untracked.string('tzeroToValidate')
    )
)
process.dqmSaver.workflow = config.workflowName
process.dqmSaver.dirName = config.outputdir

process.qTester.qtList = 'DQMOffline/CalibMuon/data/QualityTests.xml'
#process.qTester.qtList = 'DQMOffline/CalibMuon/data/QualityTests_new.xml'
