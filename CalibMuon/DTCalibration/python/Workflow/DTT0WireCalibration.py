from tools import loadCmsProcess,loadCrabCfg,loadCrabDefault,addCrabInputFile,writeCfg,prependPaths
from addPoolDBESSource import addPoolDBESSource
from CrabTask import *
import os
import shutil

class DTT0WireCalibration:
    def __init__(self, run, dir, config):
        self.pset_name = 'dtT0WireCalibration_cfg'
        self.outputROOT = 'DTTestPulses.root'
        self.outputROOTDQM = 'DQM.root'
        self.outputDB = 't0.db'
        self.tpDead = "tpDead.txt"
        self.config = config
        self.dir = dir

        self.pset_template = 'CalibMuon.DTCalibration.dtT0WireCalibration_cfg'

        self.process = None
        self.crab_cfg = None
        self.initProcess()
        self.initCrab()
        self.task = CrabTask(self.dir,self.crab_cfg)

    def initProcess(self):
        self.process = loadCmsProcess(self.pset_template)
        self.process.GlobalTag.globaltag = self.config.globaltag
        self.process.dtT0WireCalibration.rootFileName = self.outputROOT


    def initCrab(self):
        crab_cfg_parser = loadCrabCfg()
        loadCrabDefault(crab_cfg_parser,self.config)
        crab_cfg_parser.set('CMSSW','pset',self.pset_name)
        crab_cfg_parser.set('CMSSW','number_of_jobs',1)
        crab_cfg_parser.set('CMSSW','output_file','%s,%s,%s' % (self.outputDB,self.outputROOT,self.outputROOTDQM))
        if not os.path.exists(self.dir): os.makedirs(self.dir)
        shutil.copy(self.tpDead, os.path.join(self.dir,"tpDead.txt"))
        crab_cfg_parser.set('USER','additional_input_files',self.tpDead)

        self.crab_cfg = crab_cfg_parser

    def writeCfg(self):
        writeCfg(self.process,self.dir,self.pset_name)
        #writeCfgPkl(self.process,self.dir,self.pset_name)

    def run(self):
        self.project = self.task.run()
        return self.project
