from CalibMuon.DTCalibration.Workflow.addPoolDBESSource import addPoolDBESSource

class config: pass
#config.runNumber = 186323
config.runNumber = 186760
#config.t0DB = 'Run186323-dtT0WireCalibration-AbsoluteT0-Run186323-v1/Results/t0_absolute_raw_186323.db'
config.t0DB = 'Run186760-dtT0WireCalibration-AbsoluteT0-Run186760-v1/Results/t0_absolute_raw_186760.db'
config.chamberLabel = 'All'

from CalibMuon.DTCalibration.dtT0AbsoluteReferenceCorrection_cfg import process
process.source.firstRun = config.runNumber
process.dtT0AbsoluteReferenceCorrection.correctionAlgoConfig.calibChamber = config.chamberLabel
process.dtT0AbsoluteReferenceCorrection.correctionAlgoConfig.reference = 640.

process.PoolDBOutputService.connect = 'sqlite_file:t0_correction_absolute_reference_%s_%d.db' % (config.chamberLabel,config.runNumber)

addPoolDBESSource(process = process,
                  moduleName = 'calibDB',record = 'DTT0Rcd',tag = 't0',label = '',
                  connect = 'sqlite_file:%s' % config.t0DB)

#addPoolDBESSource(process = process,
#                  moduleName = 't0RefDB',record = 'DTT0Rcd',tag = config.refTag,label = config.dbLabelRef,
#                  connect = config.connect)
