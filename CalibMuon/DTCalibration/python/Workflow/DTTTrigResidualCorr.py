from tools import loadCmsProcess,writeCfg
from addPoolDBESSource import addPoolDBESSource
from CmsswTask import CmsswTask
import os

class DTTTrigResidualCorr:
    def __init__(self, run, dir, input_db, residuals, result_dir, config):
        self.runnumber = int(run)
        self.config = config
        self.dir = dir
        self.inputdb = input_db
        self.residuals = residuals
        self.result_dir = result_dir

        self.configs = ['dtTTrigResidualCorrection_cfg.py',
                        'dumpDBToFile_ResidCorr_cfg.py']

        self.pset_templates = {}
        self.pset_templates['dtTTrigResidualCorrection_cfg.py'] = 'CalibMuon.DTCalibration.dtTTrigResidualCorrection_cfg'
        self.pset_templates['dumpDBToFile_ResidCorr_cfg.py'] = 'CalibMuon.DTCalibration.dumpDBToFile_ttrig_cfg'

        self.initProcess()
        self.task = CmsswTask(self.dir,self.configs)
    
    def initProcess(self): 
        ttrig_ResidCorr = self.result_dir + '/' + 'ttrig_residuals_' + str(self.runnumber)
        ttrig_ResidCorr_db = os.path.abspath(ttrig_ResidCorr + '.db')
        ttrig_ResidCorr_txt = os.path.abspath(ttrig_ResidCorr + '.txt')
        root_file = self.residuals

        self.process = {}
        # dtTTrigResidualCorrection
        self.process['dtTTrigResidualCorrection_cfg.py'] = loadCmsProcess(self.pset_templates['dtTTrigResidualCorrection_cfg.py'])
        self.process['dtTTrigResidualCorrection_cfg.py'].source.firstRun = self.runnumber
        self.process['dtTTrigResidualCorrection_cfg.py'].GlobalTag.globaltag = self.config.globaltag

        # Input tTrig db
        if(self.inputdb):
            label = ''
            if hasattr(self.config,'runOnCosmics') and self.config.runOnCosmics: label = 'cosmics'
            addPoolDBESSource(process = self.process['dtTTrigResidualCorrection_cfg.py'],
                              moduleName = 'calibDB',record = 'DTTtrigRcd',tag = 'ttrig',label = label,
                              connect = 'sqlite_file:%s' % self.inputdb)

        # Input vDrift db
        if hasattr(self.config,'inputVdriftDB') and self.config.inputVdriftDB:
            addPoolDBESSource(process = self.process['dtTTrigResidualCorrection_cfg.py'],
                              moduleName = 'vDriftDB',record = 'DTMtimeRcd',tag = 'vDrift',
                              connect = 'sqlite_file:%s' % self.config.inputVdriftDB)

        # Change DB label if running on Cosmics
        if hasattr(self.config,'runOnCosmics') and self.config.runOnCosmics:
            self.process['dtTTrigResidualCorrection_cfg.py'].dtTTrigResidualCorrection.dbLabel = 'cosmics'
            self.process['dtTTrigResidualCorrection_cfg.py'].dtTTrigResidualCorrection.correctionAlgoConfig.dbLabel = 'cosmics'

        self.process['dtTTrigResidualCorrection_cfg.py'].PoolDBOutputService.connect = 'sqlite_file:%s' % ttrig_ResidCorr_db
        self.process['dtTTrigResidualCorrection_cfg.py'].dtTTrigResidualCorrection.correctionAlgoConfig.residualsRootFile = root_file

        # dumpDBToFile
        self.process['dumpDBToFile_ResidCorr_cfg.py'] = loadCmsProcess(self.pset_templates['dumpDBToFile_ResidCorr_cfg.py'])
        self.process['dumpDBToFile_ResidCorr_cfg.py'].calibDB.connect = 'sqlite_file:%s' % ttrig_ResidCorr_db
        self.process['dumpDBToFile_ResidCorr_cfg.py'].dumpToFile.outputFileName = ttrig_ResidCorr_txt 
 
    def writeCfg(self):
        for cfg in self.configs:
            writeCfg(self.process[cfg],self.dir,cfg)
            #writeCfgPkl(self.process[cfg],self.dir,cfg) 

    def run(self):
        self.task.run()
        return
