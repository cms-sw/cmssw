#from tools import replaceTemplate
from tools import loadCmsProcess,loadCrabCfg,loadCrabDefault,writeCfg,prependPaths
from CrabTask import *
import os

class DTTTrigValid:
    def __init__(self, run, dir, input_db, config):
        self.pset_name = 'DTkFactValidation_1_cfg.py'
        self.outputfile = 'residuals.root,DQM.root'
        self.config = config
        self.dir = dir
        self.inputdb = input_db

        self.crab_template = config.templatepath + '/crab/crab_ttrig_valid.cfg'
        self.pset_template = config.templatepath + '/config/DTkFactValidation_1_cfg.py'

        self.initProcess()
        self.initCrab()
        self.task = CrabTask(self.dir,self.crab_cfg)

    def initProcess(self):
        import FWCore.ParameterSet.Config as cms
        self.process = loadCmsProcess(self.pset_template)
        self.process.GlobalTag.globaltag = self.config.globaltag
        if(self.inputdb):
            self.process.calibDB = cms.ESSource("PoolDBESSource",self.process.CondDBSetup,
                                                            timetype = cms.string('runnumber'),
                                                            toGet = cms.VPSet(cms.PSet(
                                                                record = cms.string('DTTtrigRcd'),
                                                                tag = cms.string('ttrig')
                                                            )),
                                                            connect = cms.string('sqlite_file:'),
                                                            authenticationMethod = cms.untracked.uint32(0))

            self.process.calibDB.connect = 'sqlite_file:%s' % os.path.basename(self.inputdb)
            self.process.es_prefer_calibDB = cms.ESPrefer('PoolDBESSource','calibDB') 

        if hasattr(self.config,'inputVdriftDB') and self.config.inputVdriftDB:
            self.process.vDriftDB = cms.ESSource("PoolDBESSource",self.process.CondDBSetup,
                                                            timetype = cms.string('runnumber'),
                                                            toGet = cms.VPSet(cms.PSet(
                                                                record = cms.string('DTMtimeRcd'),
                                                                tag = cms.string('vDrift')
                                                            )),
                                                            connect = cms.string('sqlite_file:'),
                                                            authenticationMethod = cms.untracked.uint32(0))

            self.process.vDriftDB.connect = 'sqlite_file:%s' % os.path.basename(self.config.inputVdriftDB) 
            self.process.es_prefer_vDriftDB = cms.ESPrefer('PoolDBESSource','vDriftDB')

        if hasattr(self.config,'preselection') and self.config.preselection:
            pathsequence = self.config.preselection.split(':')[0]
            seqname = self.config.preselection.split(':')[1]
            self.process.load(pathsequence)
            prependPaths(self.process,seqname)

    def initCrab(self):
        crab_cfg_parser = loadCrabCfg(self.crab_template)
        loadCrabDefault(crab_cfg_parser,self.config)
        crab_cfg_parser.set('CMSSW','pset',self.pset_name)
        crab_cfg_parser.set('CMSSW','output_file',self.outputfile)
        crab_cfg_parser.remove_option('USER','additional_input_files')
        if self.inputdb: crab_cfg_parser.set('USER','additional_input_files',self.inputdb)

        if hasattr(self.config,'inputVdriftDB') and self.config.inputVdriftDB:
            additionalInputFiles = ''
            if crab_cfg_parser.has_option('USER','additional_input_files'): additionalInputFiles = crab_cfg_parser.get('USER','additional_input_files')
            if additionalInputFiles: additionalInputFiles += ',%s' % self.config.inputVdriftDB
            else: additionalInputFiles = self.config.inputVdriftDB
            crab_cfg_parser.set('USER','additional_input_files',additionalInputFiles) 
            
        
        self.crab_cfg = crab_cfg_parser

    def writeCfg(self):
        writeCfg(self.process,self.dir,self.pset_name)

    def run(self):
        self.project = self.task.run()
        return self.project
