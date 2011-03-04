from tools import loadCmsProcess,writeCfg,dqmWorkflowName
from CmsswTask import *
import os

class DTDqm:
    def __init__(self, run, dir, dqm_files, result_dir, config=None):
        self.runnumber = int(run)
        self.dir = dir
        self.result_dir = result_dir
        self.config = config
        self.dqm_files = dqm_files

        self.pset_name = 'dtDQMClient_cfg.py'
        self.pset_template = 'CalibMuon.DTCalibration.dtDQMClient_cfg'

        self.process = None
        self.initProcess()
        self.configFiles = []
        self.configFiles.append(self.pset_name)
        self.task = CmsswTask(self.dir,self.configFiles)

    def initProcess(self):
        self.process = loadCmsProcess(self.pset_template)
        self.process.source.fileNames = self.dqm_files
        self.process.dqmSaver.dirName = os.path.abspath(self.result_dir)
        if self.config:
            label = 'dtCalibration'
            if hasattr(self.config,'label') and self.config.label: label = self.config.label 
            workflowName = dqmWorkflowName(self.config.datasetpath,label,self.config.trial)
            self.process.dqmSaver.workflow = workflowName
        if self.process.DQMStore.collateHistograms: self.process.dqmSaver.forceRunNumber = self.runnumber

    def writeCfg(self):
        writeCfg(self.process,self.dir,self.pset_name)   
        #writeCfgPkl(self.process,self.dir,self.pset_name) 
    
    def run(self):
        self.task.run()
        return

def runDQM(run,castor_dir,result_dir):
    from CalibMuon.DTCalibration.Workflow.tools import listFilesInCastor
    dqm_files = listFilesInCastor(castor_dir,'DQM')
    runDir = '.'

    dtDqmFinal = DTDqm(run,runDir,dqm_files,result_dir)
    dtDqmFinal.writeCfg()
    dtDqmFinal.run()
