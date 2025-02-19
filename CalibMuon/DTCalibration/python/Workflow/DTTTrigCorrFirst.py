from tools import loadCmsProcess,writeCfg
from CmsswTask import *
import os

class DTTTrigCorrFirst:
    def __init__(self, run, dir, result_dir, config):
        #desc = 'Run%s'%run
        #desc += '/Ttrig/Exec'
        #self.desc = desc 
        self.runnumber = int(run)
        self.config = config
        self.dir = dir
        self.result_dir = result_dir

        self.configs = ['DTTTrigWriter_cfg.py','DumpDBToFile_first_cfg.py','DTTTrigCorrection_cfg.py','DumpDBToFile_second_cfg.py']

        self.pset_templates = {'DTTTrigWriter_cfg.py':config.templatepath + '/config/DTTTrigWriter_cfg.py',
                               'DumpDBToFile_first_cfg.py':config.templatepath + '/config/DumpDBToFile_ttrig_cfg.py',
                               'DTTTrigCorrection_cfg.py':config.templatepath + '/config/DTTTrigCorrection_cfg.py',
                               'DumpDBToFile_second_cfg.py':config.templatepath + '/config/DumpDBToFile_ttrig_cfg.py'}

        #self.task = CmsswTask(self.desc,self.configs,self.pset_templates,self.common_opts,self.pset_opts)
        self.initProcess()
        self.task = CmsswTask(self.dir,self.configs)

    def initProcess(self):
        timeBoxes = os.path.abspath(self.result_dir + '/' + 'DTTimeBoxes_' + str(self.runnumber) + '.root')
        ttrig_first = self.result_dir + '/' + 'ttrig_first_' + str(self.runnumber)
        ttrig_first_db = os.path.abspath(ttrig_first + '.db')
        ttrig_first_txt = os.path.abspath(ttrig_first + '.txt')
        ttrig_second = self.result_dir + '/' + 'ttrig_second_' + str(self.runnumber)
        ttrig_second_db = os.path.abspath(ttrig_second + '.db')
        ttrig_second_txt = os.path.abspath(ttrig_second + '.txt')
 
        self.process = {}
        self.process['DTTTrigWriter_cfg.py'] = loadCmsProcess(self.pset_templates['DTTTrigWriter_cfg.py'])
        self.process['DTTTrigWriter_cfg.py'].ttrigwriter.rootFileName = timeBoxes
        self.process['DTTTrigWriter_cfg.py'].PoolDBOutputService.connect = 'sqlite_file:%s' % ttrig_first_db

        self.process['DumpDBToFile_first_cfg.py'] = loadCmsProcess(self.pset_templates['DumpDBToFile_first_cfg.py'])
        self.process['DumpDBToFile_first_cfg.py'].calibDB.connect = 'sqlite_file:%s' % ttrig_first_db
        self.process['DumpDBToFile_first_cfg.py'].dumpToFile.outputFileName = ttrig_first_txt

        self.process['DTTTrigCorrection_cfg.py'] = loadCmsProcess(self.pset_templates['DTTTrigCorrection_cfg.py'])
        self.process['DTTTrigCorrection_cfg.py'].GlobalTag.globaltag = self.config.globaltag
        self.process['DTTTrigCorrection_cfg.py'].source.firstRun = self.runnumber
        self.process['DTTTrigCorrection_cfg.py'].calibDB.connect = 'sqlite_file:%s' % ttrig_first_db
        self.process['DTTTrigCorrection_cfg.py'].PoolDBOutputService.connect = 'sqlite_file:%s' % ttrig_second_db

        self.process['DumpDBToFile_second_cfg.py'] = loadCmsProcess(self.pset_templates['DumpDBToFile_second_cfg.py'])
        self.process['DumpDBToFile_second_cfg.py'].calibDB.connect = 'sqlite_file:%s' % ttrig_second_db
        self.process['DumpDBToFile_second_cfg.py'].dumpToFile.outputFileName = ttrig_second_txt  
 
    def writeCfg(self):
        for cfg in self.configs:
            writeCfg(self.process[cfg],self.dir,cfg)
           
    def run(self):
        self.task.run()

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

    dtTtrigCorrFirst = DTTTrigCorrFirst(run,config.rundir,result_dir,config)
    dtTtrigCorrFirst.writeCfg()  
    dtTtrigCorrFirst.run()

    print "Finished processing:"
    for pset in dtTtrigCorrFirst.configs: print "--->",pset
