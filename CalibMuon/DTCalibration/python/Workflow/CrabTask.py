from crabWrap import crabCreate,crabSubmit,crabWatch,getOutput
from tools import replaceTemplate
import os
#from threading import Thread

#class CrabTask(Thread):
class CrabTask:
    def __init__(self, dir, crab_cfg, pset=None, pset_name='mypset.py'):
        #Thread.__init__(self)
        self.dir = dir
  
        self.crabCfg_name = 'crab.cfg'
        self.crab_cfg = crab_cfg
  
        self.pset_name = pset_name
        self.pset = pset
        self.initializeTask(dir=self.dir)

    def initializeTask(self, dir):
        if not os.path.exists(dir): os.makedirs(dir)

        # Write pset 
        if self.pset:
            self.crab_cfg.set('CMSSW','pset',self.pset_name)
            open(dir + '/' + self.pset_name,'w').write(self.pset) 

        # Write CRAB cfg
        self.crab_cfg.write(open(dir + '/' + self.crabCfg_name,'w'))
         
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
        #self.initializeTask(dir=self.dir)
        proj = self.create(self.dir) 
        self.submit()
        return proj
