import FWCore.ParameterSet.Config as cms
from tools import loadCmsProcess,writeCfg,dqmWorkflowName,getDatasetStr
from CmsswTask import *
import os

class DTT0DBValidation:
    def __init__(self, run, dir, input_files, output_dir, config=None):
        self.runnumber = int(run)
        self.dir = dir
        self.output_dir = output_dir
        self.config = config
        self.input_files = input_files

        self.pset_template = 'DQMOffline.CalibMuon.dtT0DBValidation_cfg'

        self.configs = []
        self.initProcess()
        self.task = CmsswTask(self.dir,self.configs)

    def initProcess(self):
        refDBTag = ''
        if hasattr(self.config,'refDBTag') and self.config.refDBTag: refDBTag = self.config.refDBTag
        connect = ''
        if hasattr(self.config,'config.connectStrRefDBTag') and self.config.config.connectStrRefDBTag: connect = self.config.config.connectStrRefDBTag
        runNumbersToFiles = []
        if hasattr(self.config,'dbValidRuns') and self.config.dbValidRuns and len(self.config.dbValidRuns) == len(self.input_files): runNumbersToFiles = self.config.dbValidRuns

        self.process = {}
        idx_file = 0
        for inputFile in self.input_files:
            file = os.path.abspath(inputFile)
            fileLabel = os.path.basename(file).split('.')[0] 
            pset_name = 'dtT0DBValidation_%s_Run%d_cfg.py' % (fileLabel,self.runnumber)
            self.process[pset_name] = loadCmsProcess(self.pset_template)
	    self.process[pset_name].source.firstRun = self.runnumber

	    self.process[pset_name].tzeroRef.toGet = cms.VPSet(
		cms.PSet(
		    record = cms.string('DTT0Rcd'),
		    tag = cms.string(refDBTag),
		    label = cms.untracked.string('tzeroRef')
		    ),
		cms.PSet(
		    record = cms.string('DTT0Rcd'),
		    tag = cms.string('t0'),
		    connect = cms.untracked.string('sqlite_file:%s' % file),
		    label = cms.untracked.string('tzeroToValidate')
		    )
	        )
            self.process[pset_name].tzeroRef.connect = connect

	    if self.config:
		label = 'dtT0DBValidation'
		if hasattr(self.config,'label') and self.config.label: label = self.config.label 
		workflowName = dqmWorkflowName(self.config.datasetpath,label,self.config.trial)
                self.process[pset_name].dqmSaver.workflow = workflowName

            if runNumbersToFiles: self.process[pset_name].dqmSaver.forceRunNumber = runNumbersToFiles[idx_file]
	    self.process[pset_name].dqmSaver.dirName = os.path.abspath(self.output_dir)

            self.configs.append(pset_name)
            writeCfg(self.process[pset_name],self.dir,pset_name) 
            idx_file += 1

    """
    def writeCfg(self):
        for cfg in self.configs:
            writeCfg(self.process[cfg],self.dir,cfg)
            #writeCfgPkl(self.process,self.dir,self.pset_name) 
    """

    def run(self):
        self.task.run()
        return
