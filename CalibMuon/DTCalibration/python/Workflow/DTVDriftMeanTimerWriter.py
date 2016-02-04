from tools import loadCmsProcess,writeCfg
from addPoolDBESSource import addPoolDBESSource
from CmsswTask import CmsswTask
import os

class DTVDriftMeanTimerWriter:
    def __init__(self, run, dir, input_file, output_dir, config):
        self.runnumber = int(run)
        self.config = config
        self.dir = dir
        self.input_file = input_file
        self.output_dir = output_dir

        self.configs = ['dtVDriftMeanTimerWriter_cfg.py',
                        'dumpDBToFile_vdrift_cfg.py']

        self.pset_templates = {}
        self.pset_templates['dtVDriftMeanTimerWriter_cfg.py'] = 'CalibMuon.DTCalibration.dtVDriftMeanTimerWriter_cfg'
        self.pset_templates['dumpDBToFile_vdrift_cfg.py'] = 'CalibMuon.DTCalibration.dumpDBToFile_vdrift_cfg'

        self.initProcess()
        self.task = CmsswTask(self.dir,self.configs)
    
    def initProcess(self): 
        vDrift_meantimer = self.output_dir + '/' + 'vDrift_meantimer_' + str(self.runnumber)
        vDrift_meantimer_db = os.path.abspath(vDrift_meantimer + '.db')
        vDrift_meantimer_txt = os.path.abspath(vDrift_meantimer + '.txt')

        self.process = {}
        # dtVDriftMeanTimerWriter
        self.process['dtVDriftMeanTimerWriter_cfg.py'] = loadCmsProcess(self.pset_templates['dtVDriftMeanTimerWriter_cfg.py'])
        self.process['dtVDriftMeanTimerWriter_cfg.py'].source.firstRun = self.runnumber
        self.process['dtVDriftMeanTimerWriter_cfg.py'].GlobalTag.globaltag = self.config.globaltag

        # Input vDrift db
        if hasattr(self.config,'inputVdriftDB') and self.config.inputVdriftDB:
            addPoolDBESSource(process = self.process['dtVDriftMeanTimerWriter_cfg.py'],
                              moduleName = 'vDriftDB',record = 'DTMtimeRcd',tag = 'vDrift',
                              connect = 'sqlite_file:%s' % self.config.inputVdriftDB)

        self.process['dtVDriftMeanTimerWriter_cfg.py'].PoolDBOutputService.connect = 'sqlite_file:%s' % vDrift_meantimer_db
        self.process['dtVDriftMeanTimerWriter_cfg.py'].dtVDriftMeanTimerWriter.vDriftAlgoConfig.rootFileName = self.input_file

        # dumpDBToFile
        self.process['dumpDBToFile_vdrift_cfg.py'] = loadCmsProcess(self.pset_templates['dumpDBToFile_vdrift_cfg.py'])
        self.process['dumpDBToFile_vdrift_cfg.py'].calibDB.connect = 'sqlite_file:%s' % vDrift_meantimer_db
        self.process['dumpDBToFile_vdrift_cfg.py'].dumpToFile.outputFileName = vDrift_meantimer_txt
 
    def writeCfg(self):
        for cfg in self.configs:
            writeCfg(self.process[cfg],self.dir,cfg)
            #writeCfgPkl(self.process[cfg],self.dir,cfg) 

    def run(self):
        self.task.run()
        return
