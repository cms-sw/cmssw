from tools import replaceTemplate
from CrabTask import *
import os

class DTTTrigValid:
    def __init__(self,run,crab_opts,pset_opts,template_path):
        pset_name = 'DTkFactValidation_1_cfg.py'

        #self.crab_template = os.environ['CMSSW_BASE'] + '/src/Workflow/' + 'templates/crab/crab_Valid_TEMPL.cfg'
        #self.pset_template = os.environ['CMSSW_BASE'] + '/src/Workflow/' + 'templates/config/DTkFactValidation_1_TEMPL_cfg.py'
        self.crab_template = template_path + '/crab/crab_Valid_TEMPL.cfg'
        self.pset_template = template_path + '/config/DTkFactValidation_1_TEMPL_cfg.py'

        self.crab_opts = crab_opts
        self.crab_opts['PSET'] = pset_name

        self.pset_opts = pset_opts 

        self.crab_cfg = replaceTemplate(self.crab_template,**self.crab_opts)
        self.pset = replaceTemplate(self.pset_template,**self.pset_opts)

        desc = 'Run%s'%run
        desc += '/Ttrig/Validation'
        self.desc = desc 

        self.task = CrabTask(self.desc,self.crab_cfg,self.pset,pset_name)

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

    result_dir = 'Run%s'%run
    result_dir += '/Ttrig/Results'

    ttrig_second_db = os.path.abspath(result_dir + '/' + 'ttrig_second_' + run + '.db')

    crab_opts = {'DATASETPATH':'/StreamExpress/CRAFT09-MuAlCalIsolatedMu-v1/ALCARECO',
                 'EMAIL':'vilela@to.infn.it',
                 'RUNSELECTION':run,
                 'USERDIRCAF':'TTRIGCalibration/Validation/First/Run' + str(run) + '/v' + str(trial),
                 'INPUTFILE':ttrig_second_db}

    pset_opts = {'GLOBALTAG':'GR09_P_V1::All',
                 'INPUTFILE':ttrig_second_db.split('/')[-1]}

    dtTtrigValid = DTTTrigValid(run,crab_opts,pset_opts,'templates') 
    project = dtTtrigValid.run()

    print "Sent validation jobs with project",project
