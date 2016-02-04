#from tools import replaceTemplate
import os

class CmsswTask:
    #def __init__(self, dir, configs, pset_templates=None, common_opts=None, pset_opts=None):
    def __init__(self, dir, configs, psets=None):
        self.dir = dir  
        self.configs = configs

        #self.pset_templates = pset_templates 
        #self.common_opts = common_opts
        #self.pset_opts = pset_opts
        self.psets = psets

        self.initTask()
  
    def initTask(self):
        if self.psets:
            if not os.path.exists(self.dir): os.makedirs(self.dir)
            for pset_name in self.configs:
                pset = self.psets[pset_name]
                open(self.dir + '/' + pset_name,'w').write(pset)
 
    def run(self):
        if not os.path.exists(self.dir): os.makedirs(self.dir)
        cwd = os.getcwd()
        for pset in self.configs:
            os.chdir(self.dir)
            if not os.path.exists(pset): raise RuntimeError,'%s not found in dir %s' % (pset,os.getcwd())

            cmd = 'cmsRun %s' % pset
            print "Running", cmd, "in dir", self.dir
            os.system(cmd)
            os.chdir(cwd)       
