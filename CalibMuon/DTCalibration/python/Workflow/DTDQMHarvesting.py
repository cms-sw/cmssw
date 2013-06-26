from tools import loadCmsProcess,writeCfg,dqmWorkflowName
from addPoolDBESSource import addPoolDBESSource
from CmsswTask import *
import os

class DTDQMHarvesting:
    def __init__(self, run, dir, dqm_files, result_dir, config=None):
        self.runnumber = int(run)
        self.dir = dir
        self.result_dir = result_dir
        self.config = config
        self.dqm_files = dqm_files

        self.pset_name = 'dtDQMClient_cfg.py'
        self.pset_template = 'CalibMuon.DTCalibration.dtDQMClientAlca_cfg'

        self.process = None
        self.initProcess()
        self.configFiles = []
        self.configFiles.append(self.pset_name)
        self.task = CmsswTask(self.dir,self.configFiles)

    def initProcess(self):
        self.process = loadCmsProcess(self.pset_template)
        self.process.GlobalTag.globaltag = self.config.globaltag

	if hasattr(self.config,'inputDBTag') and self.config.inputDBTag:
	    tag = self.config.inputDBTag
	    record = self.config.inputDBRcd
	    connect = self.config.connectStrDBTag
	    moduleName = 'customDB%s' % record 
	    addPoolDBESSource(process = self.process,
			      moduleName = moduleName,record = record,tag = tag,
			      connect = connect)

        if hasattr(self.config,'inputTTrigDB') and self.config.inputTTrigDB:
            label = ''
            if hasattr(self.config,'runOnCosmics') and self.config.runOnCosmics: label = 'cosmics'
            addPoolDBESSource(process = self.process,
                              moduleName = 'tTrigDB',record = 'DTTtrigRcd',tag = 'ttrig',label = label,
                              connect = 'sqlite_file:%s' % os.path.abspath(self.config.inputTTrigDB))

        if hasattr(self.config,'inputVDriftDB') and self.config.inputVDriftDB:
            addPoolDBESSource(process = self.process,
                              moduleName = 'vDriftDB',record = 'DTMtimeRcd',tag = 'vDrift',
                              connect = 'sqlite_file:%s' % os.path.abspath(self.config.inputVDriftDB))

        if hasattr(self.config,'inputT0DB') and self.config.inputT0DB:
            addPoolDBESSource(process = self.process,
                              moduleName = 't0DB',record = 'DTT0Rcd',tag = 't0',
                              connect = 'sqlite_file:%s' % os.path.basename(self.config.inputT0DB))

        self.process.source.fileNames = self.dqm_files
        self.process.dqmSaver.dirName = os.path.abspath(self.result_dir)
        if self.config:
            label = 'dtDQMValidation'
            if hasattr(self.config,'label') and self.config.label: label = self.config.label 
            workflowName = dqmWorkflowName(self.config.datasetpath,label,self.config.trial)
            self.process.dqmSaver.workflow = workflowName
        if self.process.DQMStore.collateHistograms == True: self.process.dqmSaver.forceRunNumber = self.runnumber

    def writeCfg(self):
        writeCfg(self.process,self.dir,self.pset_name)   
        #writeCfgPkl(self.process,self.dir,self.pset_name) 
    
    def run(self):
        self.task.run()
        return

def runDQM(run,dqmFile,result_dir):
    dqm_files = [dqmFile]
    runDir = '.'

    dtDqm = DTDQMHarvesting(run,runDir,dqm_files,result_dir)
    dtDqm.writeCfg()
    dtDqm.run()
