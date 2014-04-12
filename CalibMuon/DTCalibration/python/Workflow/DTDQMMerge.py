from tools import loadCmsProcess,writeCfg,dqmWorkflowName,getDatasetStr
from CmsswTask import *
import os

class DTDQMMerge:
    def __init__(self, run, dir, dqm_files, result_dir, config=None):
        self.runnumber = int(run)
        self.dir = dir
        self.result_dir = result_dir
        self.config = config
        self.dqm_files = dqm_files

        self.pset_name = 'dtDQMMerge_cfg.py'
        self.pset_template = 'CalibMuon.DTCalibration.dtDQMMerge_cfg'

        self.process = None
        self.initProcess()
        self.configFiles = []
        self.configFiles.append(self.pset_name)
        self.task = CmsswTask(self.dir,self.configFiles)

    def initProcess(self):
        self.process = loadCmsProcess(self.pset_template)
        self.process.source.fileNames = self.dqm_files

        outputFileName = 'DQM.root'
        if self.config:
            label = 'dtDQMValidation'
            if hasattr(self.config,'label') and self.config.label: label = self.config.label 
            #workflowName = dqmWorkflowName(self.config.datasetpath,label,self.config.trial)
            datasetStr = getDatasetStr(self.config.datasetpath)
              
            outputFileName = 'DQM_%s-%s.root' % (datasetStr,label) 

        self.process.output.fileName = '%s/%s' % (os.path.abspath(self.result_dir),outputFileName) 
        
        #if self.process.DQMStore.collateHistograms:

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

    dtDqm = DTDQMMerge(run,runDir,dqm_files,result_dir)
    dtDqm.writeCfg()
    dtDqm.run()
