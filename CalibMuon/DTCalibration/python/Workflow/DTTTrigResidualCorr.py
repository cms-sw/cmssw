from tools import replaceTemplate
from CmsswTask import *
import os

class DTTTrigResidualCorr:
    def __init__(self, run, common_opts, result_dir, template_path):
        desc = 'Run%s'%run
        desc += '/Ttrig/Exec'
        self.desc = desc 

        #self.common_opts = {'GLOBALTAG':'GR09_P_V1::All'}
        self.common_opts = common_opts 

        self.configs = ['DTTTrigResidualCorrection_cfg.py','DumpDBToFile_ResidCorr_cfg.py']

        #base = os.environ['CMSSW_BASE'] + '/src/' 
        self.pset_templates = {}
        self.pset_templates['DTTTrigResidualCorrection_cfg.py'] = template_path + '/config/DTTTrigResidualCorrection_TEMPL_cfg.py'
        self.pset_templates['DumpDBToFile_ResidCorr_cfg.py'] = template_path + '/config/DumpDBToFile_ttrig_TEMPL_cfg.py'

        ttrig_second_db = os.path.abspath(result_dir + '/' + 'ttrig_second_' + run + '.db')
        ttrig_ResidCorr = result_dir + '/' + 'ttrig_ResidCorr_' + run
        ttrig_ResidCorr_db = os.path.abspath(ttrig_ResidCorr + '.db')
        ttrig_ResidCorr_txt = os.path.abspath(ttrig_ResidCorr + '.txt')
        root_file = os.path.abspath(result_dir + '/' + 'DTkFactValidation_' + run + '.root')
         
        self.pset_opts = {} 
        self.pset_opts['DTTTrigResidualCorrection_cfg.py'] = {'INPUTDBFILE':ttrig_second_db,
                                                              'OUTPUTDBFILE':ttrig_ResidCorr_db,
                                                              'INPUTROOTFILE':root_file,
                                                              'RUNNUMBER':run}
        self.pset_opts['DumpDBToFile_ResidCorr_cfg.py'] = {'INPUTFILE':ttrig_ResidCorr_db,
                                                           'OUTPUTFILE':ttrig_ResidCorr_txt}

        self.task = CmsswTask(self.desc,self.configs,self.pset_templates,self.common_opts,self.pset_opts)

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

    common_opts = {'GLOBALTAG':'GR09_P_V1::All'}

    dtTtrigResidualCorr = DTTTrigResidualCorr(run,common_opts,result_dir,'templates')  
    dtTtrigResidualCorr.run()

    print "Finished processing:"
    for pset in dtTtrigResidualCorr.configs: print "--->",pset
