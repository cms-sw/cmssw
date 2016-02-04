from tools import loadCmsProcess,writeCfg
from CmsswTask import CmsswTask
import os

class DTValidSummary:
    def __init__(self, run, dir, input_file, output_file, config):
        self.runnumber = run
        self.config = config
        self.dir = dir 
        self.input_file = input_file
        self.output_file = output_file

        self.pset_name = 'dtTTrigValidSummary_cfg.py'
        self.pset_template = 'CalibMuon.DTCalibration.dtTTrigValidSummary_cfg'

        self.process = None       
        self.initProcess()
        self.configs = [self.pset_name]
        self.task = CmsswTask(self.dir,self.configs)

    def initProcess(self):
        self.process = loadCmsProcess(self.pset_template)
        self.process.resolutionTest.inputFile = self.input_file
        self.process.resolutionTest.OutputFileName = self.output_file

    def writeCfg(self):
        writeCfg(self.process,self.dir,self.pset_name) 
        #writeCfgPkl(self.process,self.dir,self.pset_name)

    def run(self):
        self.task.run()
        return
