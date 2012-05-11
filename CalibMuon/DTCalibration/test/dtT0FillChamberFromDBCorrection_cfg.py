from CalibMuon.DTCalibration.Workflow.addPoolDBESSource import addPoolDBESSource

class config: pass
config.runNumber = 186323
#config.t0DB = 'dtT0WireCalibration-Run186323-v4/res/t0_1_1_s1L.db'
#config.t0DB = 't0_correction_chamber_Wh-2_MB2_Sec12_t0_186323.db'
config.t0DB = 't0_correction_chamber_Wh-2_MB2_Sec12_Wh2_MB3_Sec12_t0_186323.db'

config.dbLabelRef = 't0Ref'
config.refTag = 't0'
#config.connect = 'sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/Collisions11/t0/MiniDaq_Run2011A-v1_RAW/dtT0WireCalibration-Run173264-v3/Results/t0_correction_Ch_Wh2_MB1_Sec5-FillMissing_DT_t0_cosmic2009_V01_express_173264.db'
config.connect = 'sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/Collisions11/t0/MiniDaq_Run2011A-v1_RAW/dtT0WireCalibration-Run173264-v3/Results/t0_correction_Ch_Wh2_MB1_Sec5_Wh2_MB4_Sec2-FillMissing_DT_t0_cosmic2009_V01_express_173264.db'
#config.chamberLabel = 'test'
#config.chamberLabel = 'Wh-2_MB2_Sec12'
#config.chamberLabel = 'Wh2_MB3_Sec12'
config.chamberLabel = 'Wh-1_MB4_Sec1'

from CalibMuon.DTCalibration.dtT0FillChamberFromDBCorrection_cfg import process
process.source.firstRun = config.runNumber
process.dtT0FillChamberFromDBCorrection.correctionAlgoConfig.dbLabelRef = 't0Ref'
#process.dtT0FillChamberFromDBCorrection.correctionAlgoConfig.chamberId = ''
#process.dtT0FillChamberFromDBCorrection.correctionAlgoConfig.chamberId = '-2 2 12'
#process.dtT0FillChamberFromDBCorrection.correctionAlgoConfig.chamberId = '2 3 12'
process.dtT0FillChamberFromDBCorrection.correctionAlgoConfig.chamberId = '-1 4 1'
process.PoolDBOutputService.connect = 'sqlite_file:t0_correction_chamber_%s_%s_%d.db' % (config.chamberLabel,config.refTag,config.runNumber)

addPoolDBESSource(process = process,
                  moduleName = 'calibDB',record = 'DTT0Rcd',tag = 't0',label = '',
                  connect = 'sqlite_file:%s' % config.t0DB)

addPoolDBESSource(process = process,
                  moduleName = 't0RefDB',record = 'DTT0Rcd',tag = config.refTag,label = config.dbLabelRef,
                  connect = config.connect)
