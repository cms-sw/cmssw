from crabWrap import crabCreate,crabSubmit,crabWatch,getOutput
from tools import replaceTemplate
import os
#from threading import Thread

#class CrabTask(Thread):
class CrabTask:
    def __init__(self, desc, crab_cfg, pset, pset_name='pset.py'):
        #Thread.__init__(self)
        self.desc = desc
  
        self.crabCfg_name = 'crab.cfg'
        self.crab_cfg = crab_cfg
  
        self.pset_name = pset_name
        self.pset = pset
        #self.initializeTask(dir=self.desc)

    def initializeTask(self, dir):
        if not os.path.exists(dir): os.makedirs(dir)

        open(dir + '/' + self.crabCfg_name,'w').write(self.crab_cfg)
        open(dir + '/' + self.pset_name,'w').write(self.pset)

    def create(self,dir):
        self.project = crabCreate(dir,self.crabCfg_name)
        return self.project

    def submit(self):
        if not self.project: raise RuntimeError
        crabSubmit(self.project)

    def getoutput(self):
        if not self.project: raise RuntimeError
        getOutput(self.project)

    #def watch(self):
    #    if not self.project: raise RuntimeError
    #    crabWatch(getOutput,self.project) 
        
    def run(self):
        self.initializeTask(dir=self.desc)
        proj = self.create(self.desc) 
        self.submit()
        return proj

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

    #run = 100850
    #ntrial = 1 
    pset_name = 'DTTTrigCalibration_cfg.py'

    crab_template = 'templates/crab/crab_ttrig_prod_TEMPL.cfg'
    crab_opts = {'DATASETPATH':'/Cosmics/Commissioning09-v3/RAW',
                 'RUNNUMBER':run,
                 'PSET':pset_name, 
                 'USERDIRCAF':'TTRIGCalibration/Run' + str(run) + '/v' + str(trial),
                 'EMAIL':'vilela@to.infn.it'}

    pset_template = 'templates/config/DTTTrigCalibration_TEMPL_cfg.py' 
    pset_opts = {'GLOBALTAG':'CRAFT_31X::All',
                 'DIGITEMPLATE':'muonDTDigis'}

    desc = 'Run%s'%run
    desc += '/Ttrig/Production'

    crab_cfg = replaceTemplate(crab_template,**crab_opts)
    pset = replaceTemplate(pset_template,**pset_opts)

    prod = CrabTask(desc,crab_cfg,pset,pset_name)
    project = prod.run()

    print "Sent jobs with project",project
