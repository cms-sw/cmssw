from tools import loadCmsProcess,loadCrabCfg,loadCrabDefault,addCrabInputFile,writeCfg,prependPaths
from addPoolDBESSource import addPoolDBESSource
from CrabTask import *
import os

class DTVDriftSegmentCalibration:
    def __init__(self, run, dir, config):
        self.pset_name = 'dtVDriftSegmentCalibration_cfg.py'
        self.outputfile = 'DTVDriftHistos.root'
        self.config = config
        self.dir = dir

        self.pset_template = 'CalibMuon.DTCalibration.dtVDriftSegmentCalibration_cfg'
        if hasattr(self.config,'runOnCosmics') and self.config.runOnCosmics:
            self.pset_template = 'CalibMuon.DTCalibration.dtVDriftSegmentCalibration_cosmics_cfg'

        self.process = None  
        self.crab_cfg = None
        self.initProcess()
        self.initCrab()
        self.task = CrabTask(self.dir,self.crab_cfg)

    def initProcess(self):
        self.process = loadCmsProcess(self.pset_template)
        self.process.GlobalTag.globaltag = self.config.globaltag
        self.process.dtVDriftSegmentCalibration.rootFileName = self.outputfile
        # Add tTrig and vDrift DB's, if requested
        if hasattr(self.config,'inputTTrigDB') and self.config.inputTTrigDB:
            label = ''
            if hasattr(self.config,'runOnCosmics') and self.config.runOnCosmics: label = 'cosmics'
            addPoolDBESSource(process = self.process,
                              moduleName = 'tTrigDB',record = 'DTTtrigRcd',tag = 'ttrig',label = label,
                              connect = 'sqlite_file:%s' % os.path.basename(self.config.inputTTrigDB))

        if hasattr(self.config,'inputVdriftDB') and self.config.inputVdriftDB:
            addPoolDBESSource(process = self.process,
                              moduleName = 'vDriftDB',record = 'DTMtimeRcd',tag = 'vDrift',
                              connect = 'sqlite_file:%s' % os.path.basename(self.config.inputVdriftDB))

        # Prepend paths with unpacker if running on RAW
        if hasattr(self.config,'runOnRAW') and self.config.runOnRAW:
            prependPaths(self.process,self.config.digilabel)

        # Prepend paths with custom selection sequence
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
        if hasattr(self.config,'inputTTrigDB') and self.config.inputTTrigDB:
            addCrabInputFile(crab_cfg_parser,self.config.inputTTrigDB)

        if hasattr(self.config,'inputVdriftDB') and self.config.inputVdriftDB:
            addCrabInputFile(crab_cfg_parser,self.config.inputVdriftDB)

        self.crab_cfg = crab_cfg_parser

    def writeCfg(self):
        writeCfg(self.process,self.dir,self.pset_name)
        #writeCfgPkl(self.process,self.dir,self.pset_name)

    def run(self):
        self.project = self.task.run()
        return self.project
