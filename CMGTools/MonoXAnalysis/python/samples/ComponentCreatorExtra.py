import PhysicsTools.HeppyCore.framework.config as cfg
from CMGTools.Production import eostools
from CMGTools.Production.dataset import createDataset, createMyDataset
from CMGTools.TTHAnalysis.samples.ComponentCreator import ComponentCreator
import re

class ComponentCreatorExtra(object):
    def makePrivateDataComponent(self,name,dataset,files,json,xSec=1):
         if len(files) == 0:
            raise RuntimeError, "Trying to make a component %s with no files" % name
         # prefix filenames with dataset unless they start with "/"
         dprefix = dataset +"/" if files[0][0] != "/" else ""
         component = cfg.DataComponent(
             name = name,
             files = ['root://eoscms.cern.ch//eos/cms%s%s' % (dprefix,f) for f in files],
             intLumi=1,
             triggers = [],
             json=json
         )

         return component
