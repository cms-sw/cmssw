from tools import loadCmsProcess,loadCrabCfg,loadCrabDefault,addCrabInputFile,writeCfg,prependPaths
from addPoolDBESSource import addPoolDBESSource
from CrabTask import *
import os

class DTTTrigValid:
    def __init__(self, run, dir, input_db, config):
        self.pset_name = 'dtCalibValidation_cfg.py'
        #self.outputfile = 'residuals.root,DQM.root'
        self.outputfile = 'DQM.root'
        self.config = config
        self.dir = dir
        self.inputdb = input_db

        self.pset_template = 'CalibMuon.DTCalibration.dtCalibValidation_cfg'
        if hasattr(self.config,'runOnCosmics') and self.config.runOnCosmics:
            self.pset_template = 'CalibMuon.DTCalibration.dtCalibValidation_cosmics_cfg'

        self.process = None  
        self.crab_cfg = None
        self.initProcess()
        self.initCrab()
        self.task = CrabTask(self.dir,self.crab_cfg)

    def initProcess(self):
        self.process = loadCmsProcess(self.pset_template)
        self.process.GlobalTag.globaltag = self.config.globaltag
        #self.process.dtCalibValidation.OutputMEsInRootFile = True

        if(self.inputdb):
            label = ''
            if hasattr(self.config,'runOnCosmics') and self.config.runOnCosmics: label = 'cosmics'
            addPoolDBESSource(process = self.process,
                              moduleName = 'calibDB',record = 'DTTtrigRcd',tag = 'ttrig',label=label,
                              connect = 'sqlite_file:%s' % os.path.basename(self.inputdb))

        if hasattr(self.config,'inputVdriftDB') and self.config.inputVdriftDB:
            addPoolDBESSource(process = self.process,
                              moduleName = 'vDriftDB',record = 'DTMtimeRcd',tag = 'vDrift',
                              connect = 'sqlite_file:%s' % os.path.basename(self.config.inputVdriftDB))

        if hasattr(self.config,'runOnRAW') and self.config.runOnRAW:
            prependPaths(self.process,self.config.digilabel)
 
        if hasattr(self.config,'preselection') and self.config.preselection:
            pathsequence = self.config.preselection.split(':')[0]
            seqname = self.config.preselection.split(':')[1]
            self.process.load(pathsequence)
            prependPaths(self.process,seqname)

    def initCrab(self):
        crab_cfg_parser = loadCrabCfg()
        loadCrabDefault(crab_cfg_parser,self.config)
        crab_cfg_parser.set('CMSSW','pset',self.pset_name)
        crab_cfg_parser.set('CMSSW','output_file',self.outputfile)
        crab_cfg_parser.remove_option('USER','additional_input_files')
        if self.inputdb:
            addCrabInputFile(crab_cfg_parser,self.inputdb)

        if hasattr(self.config,'inputVdriftDB') and self.config.inputVdriftDB:
            addCrabInputFile(crab_cfg_parser,self.config.inputVdriftDB)

        self.crab_cfg = crab_cfg_parser

    def writeCfg(self):
        writeCfg(self.process,self.dir,self.pset_name)
        #writeCfgPkl(self.process,self.dir,self.pset_name)

    def run(self):
        self.project = self.task.run()
        return self.project
