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

        #self.crab_template = os.environ['CMSSW_BASE'] + '/src/Workflow/' + 'templates/crab/crab_Valid_TEMPL.cfg'
        #self.pset_template = os.environ['CMSSW_BASE'] + '/src/Workflow/' + 'templates/config/DTkFactValidation_1_TEMPL_cfg.py'
        self.crab_template = config.templatepath + '/crab/crab_ttrig_valid.cfg'
        self.pset_template = config.templatepath + '/config/DTkFactValidation_1_cfg.py'

        #self.crab_opts = crab_opts
        #self.crab_opts['PSET'] = pset_name

        #self.pset_opts = pset_opts 

        #self.crab_cfg = replaceTemplate(self.crab_template,**self.crab_opts)
        #self.pset = replaceTemplate(self.pset_template,**self.pset_opts)

        #desc = 'Run%s'%run
        #desc += '/Ttrig/Validation'
        #self.desc = desc 

        self.initProcess()
        self.initCrab()
        #self.task = CrabTask(self.desc,self.crab_cfg,self.pset,pset_name)
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
        if self.inputdb: crab_cfg_parser.set('USER','additional_input_files',self.inputdb)
        else: crab_cfg_parser.remove_option('USER','additional_input_files')
        self.crab_cfg = crab_cfg_parser

    def writeCfg(self):
        writeCfg(self.process,self.dir,self.pset_name)

    def run(self):
        self.project = self.task.run()
        return self.project

if __name__ == '__main__':

    run = None
    trial = None
    import sys
    for opt in sys.argv:
        if opt[:4] == 'run=':
            run = opt[4:]
        if opt[:6] == 'trial=':
            trial = opt[6:]

    if not run: raise ValueError,'Need to set run number'
    if not trial: raise ValueError,'Need to set trial number'

    run_dir = 'Test'
    ttrig_second_db = os.path.abspath(run_dir + '/' + 'ttrig_second_' + run + '.db')

    config.globaltag = 'GR09_P_V1::All'
    config.scheduler = 'CAF'
    config.useserver = True
    config.datasetpath = '/StreamExpress/CRAFT09-MuAlCalIsolatedMu-v1/ALCARECO'
    config.runselection = run
    config.totalnumberevents = 1000000
    config.eventsperjob = 50000
    config.stageOutCAF = True
    config.userdircaf = 'TTRIGCalibration/Validation/First/Run' + str(run) + '/v' + str(trial)
    config.email = 'vilela@to.infn.it'
    config.templatepath = 'templates'

    dtTtrigValid = DTTTrigValid(run,run_dir,ttrig_second_db,config) 
    #project = dtTtrigValid.run()
    #print "Sent validation jobs with project",project
