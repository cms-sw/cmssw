from tools import loadCmsProcess,writeCfg
from CmsswTask import *
import os

class DTDqm:
    def __init__(self, run, dir, dqm_files, result_dir, template_path):
        #basedir = 'Run%s/Ttrig' % run
        #self.dir = basedir + '/' + 'Exec'
        #self.result_dir = basedir + '/' + 'Results'
        self.run = run
        self.dir = dir
        self.result_dir = result_dir
        self.dqm_files = dqm_files

        self.pset_name = 'DTkFactValidation_2_DQM_cfg.py'
        self.pset_template = template_path + '/config/DTkFactValidation_2_DQM_cfg.py'

        self.initProcess()
        configs = []
        configs.append(self.pset_name)
        self.task = CmsswTask(self.dir,configs)

    def initProcess(self):
        self.process = loadCmsProcess(self.pset_template)
        self.process.source.fileNames = self.dqm_files
        self.process.dqmSaver.dirName = os.path.abspath(self.result_dir)
        if self.process.DQMStore.collateHistograms: self.process.dqmSaver.forceRunNumber = self.run 

    def writeCfg(self):
        writeCfg(self.process,self.dir,self.pset_name) 
    
    def run(self):
        self.task.run()
        return

def runDQM(run,castor_dir,result_dir,template_path):
    from CalibMuon.DTCalibration.Workflow.tools import listFilesInCastor
    dqm_files = listFilesInCastor(castor_dir,'DQM')
    runDir = '.'
    dtDqmFinal = DTDqm(run,runDir,dqm_files,result_dir,template_path)
    dtDqmFinal.writeCfg()
    dtDqmFinal.run()
