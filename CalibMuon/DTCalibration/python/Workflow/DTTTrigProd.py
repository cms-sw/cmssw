from tools import loadCmsProcess,loadCrabCfg,loadCrabDefault,writeCfg,prependPaths
from CrabTask import *
import os

class DTTTrigProd:
    def __init__(self, run, dir, config):
        self.pset_name = 'dtTTrigCalibration_cfg.py'
        self.outputfile = 'DTTimeBoxes.root'
        self.config = config
        self.dir = dir 
        self.pset_template = 'CalibMuon.DTCalibration.dtTTrigCalibration_cfg'
        self.process = None
        self.crab_cfg = None

        self.initProcess()
        self.initCrab()
        self.task = CrabTask(self.dir,self.crab_cfg) 

    def initProcess(self):
        self.process = loadCmsProcess(self.pset_template)
        self.process.GlobalTag.globaltag = self.config.globaltag
        self.process.dtTTrigCalibration.rootFileName = self.outputfile
        self.process.dtTTrigCalibration.digiLabel = self.config.digilabel
        if hasattr(self.config,'runOnCosmics') and self.config.runOnCosmics:
            self.process.load('RecoLocalMuon.Configuration.RecoLocalMuonCosmics_cff')

        if hasattr(self.config,'runOnRAW') and self.config.runOnRAW:
            prependPaths(self.process,self.config.digilabel)

        if hasattr(self.config,'preselection') and self.config.preselection:
            pathsequence = self.config.preselection.split(':')[0]
            seqname = self.config.preselection.split(':')[1]
            self.process.load(pathsequence)
            prependPaths(self.process,seqname)

    def initCrab(self):
        crab_cfg_parser = loadCrabCfg()
        crab_cfg_parser = loadCrabDefault(crab_cfg_parser,self.config)
        crab_cfg_parser.set('CMSSW','pset',self.pset_name)
        crab_cfg_parser.set('CMSSW','output_file',self.outputfile) 
        self.crab_cfg = crab_cfg_parser

    def writeCfg(self):
        writeCfg(self.process,self.dir,self.pset_name)
        #writeCfgPkl(self.process,self.dir,self.pset_name) 

    def run(self):
        self.project = self.task.run() 
        return self.project
