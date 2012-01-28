from tools import loadCmsProcess,writeCfg
from addPoolDBESSource import addPoolDBESSource
from CmsswTask import CmsswTask
import os

class DTVDriftSegmentWriter:
    def __init__(self, run, dir, input_file, output_dir, config):
        self.runnumber = int(run)
        self.config = config
        self.dir = dir
        self.input_file = input_file
        self.output_dir = output_dir

        self.configs = ['dtVDriftSegmentWriter_cfg.py',
                        'dumpDBToFile_vdrift_cfg.py']

        self.pset_templates = {}
        self.pset_templates['dtVDriftSegmentWriter_cfg.py'] = 'CalibMuon.DTCalibration.dtVDriftSegmentWriter_cfg'
        self.pset_templates['dumpDBToFile_vdrift_cfg.py'] = 'CalibMuon.DTCalibration.dumpDBToFile_vdrift_cfg'

        self.initProcess()
        self.task = CmsswTask(self.dir,self.configs)
    
    def initProcess(self): 
        vDrift_segment = self.output_dir + '/' + 'vDrift_segment_' + str(self.runnumber)
        vDrift_segment_db = os.path.abspath(vDrift_segment + '.db')
        vDrift_segment_txt = os.path.abspath(vDrift_segment + '.txt')

        self.process = {}
        # dtVDriftSegmentWriter
        self.process['dtVDriftSegmentWriter_cfg.py'] = loadCmsProcess(self.pset_templates['dtVDriftSegmentWriter_cfg.py'])
        self.process['dtVDriftSegmentWriter_cfg.py'].source.firstRun = self.runnumber
        self.process['dtVDriftSegmentWriter_cfg.py'].GlobalTag.globaltag = self.config.globaltag

        # Input vDrift db
        if hasattr(self.config,'inputVdriftDB') and self.config.inputVdriftDB:
            addPoolDBESSource(process = self.process['dtVDriftSegmentWriter_cfg.py'],
                              moduleName = 'vDriftDB',record = 'DTMtimeRcd',tag = 'vDrift',
                              connect = 'sqlite_file:%s' % self.config.inputVdriftDB)

        self.process['dtVDriftSegmentWriter_cfg.py'].PoolDBOutputService.connect = 'sqlite_file:%s' % vDrift_segment_db
        self.process['dtVDriftSegmentWriter_cfg.py'].dtVDriftSegmentWriter.vDriftAlgoConfig.rootFileName = self.input_file

        # dumpDBToFile
        self.process['dumpDBToFile_vdrift_cfg.py'] = loadCmsProcess(self.pset_templates['dumpDBToFile_vdrift_cfg.py'])
        self.process['dumpDBToFile_vdrift_cfg.py'].calibDB.connect = 'sqlite_file:%s' % vDrift_segment_db
        self.process['dumpDBToFile_vdrift_cfg.py'].dumpToFile.outputFileName = vDrift_segment_txt
 
    def writeCfg(self):
        for cfg in self.configs:
            writeCfg(self.process[cfg],self.dir,cfg)
            #writeCfgPkl(self.process[cfg],self.dir,cfg) 

    def run(self):
        self.task.run()
        return
