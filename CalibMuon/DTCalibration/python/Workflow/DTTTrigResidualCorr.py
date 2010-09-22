from tools import loadCmsProcess,writeCfg
from CmsswTask import *
import os

class DTTTrigResidualCorr:
    def __init__(self, run, dir, input_db, residuals, result_dir, config):
        self.runnumber = int(run)
        self.config = config
        self.dir = dir
        self.inputdb = input_db
        self.residuals = residuals
        self.result_dir = result_dir


        self.configs = ['DTTTrigResidualCorrection_cfg.py','DumpDBToFile_ResidCorr_cfg.py']

        self.pset_templates = {}
        self.pset_templates['DTTTrigResidualCorrection_cfg.py'] = config.templatepath + '/config/DTTTrigResidualCorrection_cfg.py'
        self.pset_templates['DumpDBToFile_ResidCorr_cfg.py'] = config.templatepath + '/config/DumpDBToFile_ttrig_cfg.py'

        self.initProcess()
        self.task = CmsswTask(self.dir,self.configs)
    
    def initProcess(self): 
        import FWCore.ParameterSet.Config as cms
        ttrig_ResidCorr = self.result_dir + '/' + 'ttrig_ResidCorr_' + str(self.runnumber)
        ttrig_ResidCorr_db = os.path.abspath(ttrig_ResidCorr + '.db')
        ttrig_ResidCorr_txt = os.path.abspath(ttrig_ResidCorr + '.txt')
        #root_file = os.path.abspath(self.result_dir + '/' + 'DTkFactValidation_' + str(self.runnumber) + '.root')
        root_file = self.residuals

        self.process = {}
        # DTTTrigResidualCorrection
        self.process['DTTTrigResidualCorrection_cfg.py'] = loadCmsProcess(self.pset_templates['DTTTrigResidualCorrection_cfg.py'])
        self.process['DTTTrigResidualCorrection_cfg.py'].source.firstRun = self.runnumber
        self.process['DTTTrigResidualCorrection_cfg.py'].GlobalTag.globaltag = self.config.globaltag
        # Input tTrig db
        if(self.inputdb):
            condDBSetup = self.process['DTTTrigResidualCorrection_cfg.py'].CondDBSetup 
            self.process['DTTTrigResidualCorrection_cfg.py'].calibDB = cms.ESSource("PoolDBESSource",
                                                            condDBSetup,
                                                            timetype = cms.string('runnumber'),
                                                            toGet = cms.VPSet(cms.PSet(
                                                                record = cms.string('DTTtrigRcd'),
                                                                tag = cms.string('ttrig')
                                                            )),
                                                            connect = cms.string('sqlite_file:'),
                                                            authenticationMethod = cms.untracked.uint32(0))

            self.process['DTTTrigResidualCorrection_cfg.py'].calibDB.connect = 'sqlite_file:%s' % self.inputdb
            self.process['DTTTrigResidualCorrection_cfg.py'].es_prefer_calibDB = cms.ESPrefer('PoolDBESSource','calibDB')

        # Input vDrift db
        if hasattr(self.config,'inputVdriftDB') and self.config.inputVdriftDB:
            condDBSetup = self.process['DTTTrigResidualCorrection_cfg.py'].CondDBSetup
            self.process['DTTTrigResidualCorrection_cfg.py'].vDriftDB = cms.ESSource("PoolDBESSource",condDBSetup,
                                                            timetype = cms.string('runnumber'),
                                                            toGet = cms.VPSet(cms.PSet(
                                                                record = cms.string('DTMtimeRcd'),
                                                                tag = cms.string('vDrift')
                                                            )),
                                                            connect = cms.string('sqlite_file:'),
                                                            authenticationMethod = cms.untracked.uint32(0))

            self.process['DTTTrigResidualCorrection_cfg.py'].vDriftDB.connect = 'sqlite_file:%s' % self.config.inputVdriftDB
            self.process['DTTTrigResidualCorrection_cfg.py'].es_prefer_vDriftDB = cms.ESPrefer('PoolDBESSource','vDriftDB')
 
        self.process['DTTTrigResidualCorrection_cfg.py'].PoolDBOutputService.connect = 'sqlite_file:%s' % ttrig_ResidCorr_db
        self.process['DTTTrigResidualCorrection_cfg.py'].DTTTrigCorrection.correctionAlgoConfig.residualsRootFile = root_file

        # DumpDBToFile
        self.process['DumpDBToFile_ResidCorr_cfg.py'] = loadCmsProcess(self.pset_templates['DumpDBToFile_ResidCorr_cfg.py'])
        self.process['DumpDBToFile_ResidCorr_cfg.py'].calibDB.connect = 'sqlite_file:%s' % ttrig_ResidCorr_db
        self.process['DumpDBToFile_ResidCorr_cfg.py'].dumpToFile.outputFileName = ttrig_ResidCorr_txt 
 
    def writeCfg(self):
        for cfg in self.configs:
            writeCfg(self.process[cfg],self.dir,cfg)

    def run(self):
        self.task.run()
        return

if __name__ == '__main__':

    run = None
    import sys
    for opt in sys.argv:
        if opt[:4] == 'run=':
            run = opt[4:] 
 
    if not run: raise ValueError,'Need to set run number' 
 
    result_dir = 'Run%s'%run
    result_dir += '/Ttrig/Results'
    if not os.path.exists(result_dir): os.makedirs(result_dir)

    class config: pass
    config.templatepath = 'templates'
    config.globaltag = 'GR09_P_V1::All'
    config.rundir = '.'

    dtTtrigResidualCorr = DTTTrigResidualCorr(run,config.rundir,result_dir,config)  
    dtTtrigResidualCorr.run()

    print "Finished processing:"
    for pset in dtTtrigResidualCorr.configs: print "--->",pset
