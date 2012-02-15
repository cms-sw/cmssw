from tools import loadCmsProcess,loadCrabCfg,loadCrabDefault,addCrabInputFile,writeCfg,prependPaths
from addPoolDBESSource import addPoolDBESSource
from CrabTask import *
import os

class DTNoiseCalibration:
    def __init__(self, run, dir, config):
        self.pset_name = 'dtNoiseCalibration_cfg.py'
        self.outputROOT = 'dtNoiseCalib.root'
        self.outputDB = 'noise.db'
        self.config = config
        self.dir = dir

        self.pset_template = 'CalibMuon.DTCalibration.dtNoiseCalibration_cfg'
        #if hasattr(self.config,'runOnCosmics') and self.config.runOnCosmics:
        #    self.pset_template = 'CalibMuon.DTCalibration.dtNoiseCalibration_cosmics_cfg'

        self.process = None  
        self.crab_cfg = None
        self.initProcess()
        self.initCrab()
        self.task = CrabTask(self.dir,self.crab_cfg)

    def initProcess(self):
        self.process = loadCmsProcess(self.pset_template)
        self.process.GlobalTag.globaltag = self.config.globaltag
        self.process.dtNoiseCalibration.rootFileName = self.outputROOT 

	if hasattr(self.config,'inputDBTag') and self.config.inputDBTag:
	    tag = self.config.inputDBTag
	    record = self.config.inputDBRcd
	    connect = self.config.connectStrDBTag
	    moduleName = 'customDB%s' % record 
	    addPoolDBESSource(process = self.process,
			      moduleName = moduleName,record = record,tag = tag,
			      connect = connect)

        if hasattr(self.config,'runOnRAW') and self.config.runOnRAW:
            if hasattr(self.config,'runOnMC') and self.config.runOnMC:
                getattr(self.process,self.config.digilabel).inputLabel = 'rawDataCollector' 
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
        crab_cfg_parser.set('CMSSW','output_file','%s,%s' % (self.outputDB,self.outputROOT))
        crab_cfg_parser.remove_option('USER','additional_input_files')

        self.crab_cfg = crab_cfg_parser

    def writeCfg(self):
        writeCfg(self.process,self.dir,self.pset_name)
        #writeCfgPkl(self.process,self.dir,self.pset_name)

    def run(self):
        self.project = self.task.run()
        return self.project
