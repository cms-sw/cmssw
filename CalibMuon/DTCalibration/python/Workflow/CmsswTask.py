from tools import replaceTemplate
import os

class CmsswTask:
    def __init__(self, desc, configs, common_opts, pset_templates, pset_opts):
        self.desc = desc
        self.dir = desc  
        self.configs = configs
        self.common_opts = common_opts
        self.pset_templates = pset_templates
        self.pset_opts = pset_opts

        self.pset_list = None
        self.initTask()
  
    def initTask(self):
        if not os.path.exists(self.dir): os.makedirs(self.dir)

        #pset_list = []
        for pset_name in self.configs:
            opts = self.pset_opts[pset_name]
            opts.update(self.common_opts)
            pset = replaceTemplate(self.pset_templates[pset_name],**opts)
            open(self.dir + '/' + pset_name,'w').write(pset)
 
        #self.pset_list = pset_list

    def run(self):

        cwd = os.getcwd()
        for pset in self.configs:
            os.chdir(self.dir)
            cmd = 'cmsRun %s'%pset
            print "Running", cmd, "in dir", self.dir
            os.system(cmd)
            os.chdir(cwd)       
     
if __name__ == '__main__':

    run = None
    import sys
    for opt in sys.argv:
        if opt[:4] == 'run=':
            run = opt[4:] 
 
    if not run: raise ValueError,'Need to set run number' 
 
    desc = 'Run%s'%run
    desc += '/Ttrig/Exec'

    common_opts = {'GLOBALTAG':'CRAFT_31X::All'}

    configs = ['DTTTrigWriter_cfg.py','DumpDBToFile_first_cfg.py','DTTTrigCorrection_cfg.py','DumpDBToFile_second_cfg.py']
 
    pset_templates = {'DTTTrigWriter_cfg.py':'Workflow/templates/config/DTTTrigWriter_TEMPL_cfg.py',
                      'DumpDBToFile_first_cfg.py':'Workflow/templates/config/DumpDBToFile_ttrig_TEMPL_cfg.py',
                      'DTTTrigCorrection_cfg.py':'Workflow/templates/config/DTTTrigCorrection_TEMPL_cfg.py',
                      'DumpDBToFile_second_cfg.py':'Workflow/templates/config/DumpDBToFile_ttrig_TEMPL_cfg.py'}

    result_dir = 'Run%s'%run
    result_dir += '/Ttrig/Results'
    if not os.path.exists(result_dir): os.makedirs(result_dir)

    timeBoxes = os.path.abspath(result_dir + '/' + 'DTTimeBoxes_' + run + '.root')
    ttrig_first = result_dir + '/' + 'ttrig_first_' + run
    ttrig_first_db = os.path.abspath(ttrig_first + '.db')
    ttrig_first_txt = os.path.abspath(ttrig_first + '.txt')
    ttrig_second = result_dir + '/' + 'ttrig_second_' + run
    ttrig_second_db = os.path.abspath(ttrig_second + '.db')
    ttrig_second_txt = os.path.abspath(ttrig_second + '.txt')

    pset_opts = {'DTTTrigWriter_cfg.py':{'INPUTFILE':timeBoxes,'OUTPUTFILE':ttrig_first_db},
                 'DumpDBToFile_first_cfg.py':{'INPUTFILE':ttrig_first_db,'OUTPUTFILE':ttrig_first_txt},
                 'DTTTrigCorrection_cfg.py':{'INPUTFILE':ttrig_first_db,'OUTPUTFILE':ttrig_second_db},
                 'DumpDBToFile_second_cfg.py':{'INPUTFILE':ttrig_second_db,'OUTPUTFILE':ttrig_second_txt}}

    task = CmsswTask(desc,configs,common_opts,pset_templates,pset_opts)
    task.run()

    print "Finished processing:"
    for pset in configs: print "--->",pset
