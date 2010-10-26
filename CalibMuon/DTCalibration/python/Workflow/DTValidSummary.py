from tools import loadCmsProcess,writeCfg
from CmsswTask import *
import os

class DTValidSummary:
    def __init__(self, dir, input_file, output_file, template_path):
        self.dir = dir 
        self.input_file = input_file
        self.output_file = output_file

        self.pset_name = 'DTkFactValidation_2_cfg.py'
        self.pset_template = template_path + '/config/DTkFactValidation_2_cfg.py'

        self.initProcess()
        configs = [self.pset_name]
        self.task = CmsswTask(self.dir,configs)

    def initProcess(self):
        self.process = loadCmsProcess(self.pset_template)
        self.process.resolutionTest.inputFile = self.input_file
        self.process.resolutionTest.OutputFileName = self.output_file

    def writeCfg(self):
        writeCfg(self.process,self.dir,self.pset_name) 

    def run(self):
        self.task.run()
        return
