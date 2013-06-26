from CalibMuon.DTCalibration.Workflow.addPoolDBESSource import addPoolDBESSource

from CalibMuon.DTCalibration.dtT0Analyzer_cfg import process
process.dtT0Analyzer.rootFileName = "dtT0Analyzer_dtT0WireCalibration-Run186323-v4.root"

addPoolDBESSource(process = process,
                  moduleName = 'calibDB',record = 'DTT0Rcd',tag = 't0',label = '',
                  connect = 'sqlite_file:dtT0WireCalibration-Run186323-v4/res/t0_1_1_s1L.db')
