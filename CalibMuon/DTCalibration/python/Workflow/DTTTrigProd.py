#from tools import replaceTemplate
from tools import loadCmsProcess,loadCrabCfg,loadCrabDefault,writeCfg,prependPaths
from CrabTask import *
import os

class DTTTrigProd:
    #def __init__(self, run, crab_opts, pset_opts, template_path):
    def __init__(self, run, dir, config):
        self.pset_name = 'DTTTrigCalibration_cfg.py'
        self.outputfile = 'DTTimeBoxes.root'
        self.config = config
        self.dir = dir 

        #self.crab_template = os.environ['CMSSW_BASE'] + '/src/Workflow/' + 'templates/crab/crab_ttrig_prod_TEMPL.cfg'
        #self.pset_template = os.environ['CMSSW_BASE'] + '/src/Workflow/' + 'templates/config/DTTTrigCalibration_TEMPL_cfg.py'
        self.crab_template = config.templatepath + '/crab/crab_ttrig_prod.cfg'
        self.pset_template = config.templatepath + '/config/DTTTrigCalibration_cfg.py'

        #self.crab_opts = crab_opts
        #self.crab_opts['PSET'] = pset_name
        #self.pset_opts = pset_opts

        #self.crab_cfg = replaceTemplate(self.crab_template,**self.crab_opts)
        #self.pset = replaceTemplate(self.pset_template,**self.pset_opts)

        #desc = 'Run%s'%run
        #desc += '/Ttrig/Production'
        #self.desc = desc 

        self.initProcess()
        self.initCrab()
        #self.task = CrabTask(self.dir,self.crab_cfg,self.pset,self.pset_name)
        self.task = CrabTask(self.dir,self.crab_cfg) 

    def initProcess(self):
        self.process = loadCmsProcess(self.pset_template)
        self.process.GlobalTag.globaltag = self.config.globaltag
        self.process.ttrigcalib.digiLabel = self.config.digilabel
        if hasattr(self.config,'preselection') and self.config.preselection:
            pathsequence = self.config.preselection.split(':')[0]
            seqname = self.config.preselection.split(':')[1]
            self.process.load(pathsequence)
            prependPaths(self.process,seqname)

        #writeCfg(self.process,self.dir,self.pset_name)
        #self.pset = self.process.dumpPython()

    def initCrab(self):
        crab_cfg_parser = loadCrabCfg(self.crab_template)
        loadCrabDefault(crab_cfg_parser,self.config)
        crab_cfg_parser.set('CMSSW','pset',self.pset_name)
        crab_cfg_parser.set('CMSSW','output_file',self.outputfile) 
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

    class config: pass

    config.globaltag = 'GR09_P_V1::All'
    config.digilabel = 'muonDTDigis'
    
    config.scheduler = 'CAF'
    config.useserver = False
    config.datasetpath = '/StreamExpress/CRAFT09-MuAlCalIsolatedMu-v1/ALCARECO'
    config.runselection = run
    config.totalnumberevents = 1000000
    config.eventsperjob = 50000
    config.stageOutCAF = True
    config.userdircaf = 'TTRIGCalibration/Production/Run' + str(run) + '/v' + str(trial)
    config.email = 'vilela@to.infn.it'

    dtTtrigProd = DTTTrigProd(run,'Test',config) 
    #project = dtTtrigProd.run()
    #print "Sent production jobs with project",project
