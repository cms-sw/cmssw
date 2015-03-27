#definition of pythonized VIDElectron class
import os
__path__.append(os.path.dirname(os.path.abspath(__file__).rsplit('/RecoEgamma/ElectronIdentification/',1)[0])+'/cfipython/slc6_amd64_gcc491/RecoEgamma/ElectronIdentification')

import ROOT

# load FWLite C++ libraries
ROOT.gSystem.Load("libFWCoreFWLite.so");
ROOT.gSystem.Load("libDataFormatsFWLite.so");
ROOT.AutoLibraryLoader.enable()

#load versioned ID selector
ROOT.gSystem.Load("libFWCorePythonParameterSet.so")
ROOT.gSystem.Load("libFWCoreParameterSet.so")
ROOT.gSystem.Load("libRecoEgammaElectronIdentification.so")

VersionedGsfElectronSelector = ROOT.VersionedGsfElectronSelector

config_template = """
import FWCore.ParameterSet.Config as cms

from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

ebCutOff = 1.479

%s
"""

def noodle( pythonpset ):    
    escaped_pset = config_template%(pythonpset)
    escaped_pset = escaped_pset.replace('"',"'")
    escaped_pset = escaped_pset.replace('\n','\\n')    
    
    idname = str(pythonpset.idName).replace('-','_')
    idname = idname.replace("cms.string('","")
    idname = idname.replace("')","")
    
    ROOT.gROOT.ProcessLine('#include "FWCore/PythonParameterSet/interface/MakeParameterSets.h"')
    ROOT.gROOT.ProcessLine('const edm::ParameterSet& %s = edm::readPSetsFrom("%s")->getParameter<edm::ParameterSet>("%s");'%(idname,escaped_pset,idname))
    
    return getattr(ROOT,idname)

class VIDElectronSelector:
    def __init__(self,pythonpset = None):        
        self.initialized_ = False
        self.instance_ = None
        if pythonpset is not None:
            config = noodle(pythonpset)
            self.instance_ = VersionedGsfElectronSelector(config)
            self.initialized_ = True
        else:
            self.instance_ = VersionedGsfElectronSelector()
    
    def __call__(self,*args):
        return self.instance_(*args)
        
    def initialize(self,pythonpset):
        if( self.initialized_ ): return
        config = noodle(pythonpset)         
        self.instance_.initialize(config)
        self.initialized_ = True

    def cutFlowSize(self):
        return self.instance_.cutFlowSize()

    def howFarInCutFlow(self):
        return self.instance_.howFarInCutFlow()

    def name(self):
        return self.instance_.name()

    def md5String(self):
        return self.instance_.md5String()

    def md55Raw(self):
        return self.instance_.md55Raw()

    
