#definition of pythonized VIDElectron class
import os
__path__.append(os.path.dirname(os.path.abspath(__file__).rsplit('/RecoEgamma/ElectronIdentification/',1)[0])+'/cfipython/slc6_amd64_gcc491/RecoEgamma/ElectronIdentification')

import ROOT
import string
import random
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


# load FWLite C++ libraries
ROOT.gSystem.Load("libFWCoreFWLite.so");
ROOT.gSystem.Load("libDataFormatsFWLite.so");
ROOT.AutoLibraryLoader.enable()

#load versioned ID selector
ROOT.gSystem.Load("libFWCorePythonParameterSet.so")
ROOT.gSystem.Load("libFWCoreParameterSet.so")
ROOT.gSystem.Load("libRecoEgammaElectronIdentification.so")

VersionedGsfElectronSelector = ROOT.VersionedGsfElectronSelector
MakeVersionedSelector = ROOT.MakeVersionedSelector
MakeGsfVersionedSelector = MakeVersionedSelector(ROOT.reco.GsfElectron)


config_template = """
import FWCore.ParameterSet.Config as cms

from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

ebCutOff = 1.479

%s
"""

#define some convenience functions in C++
ROOT.gROOT.ProcessLine("""
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

edm::Ptr<reco::GsfElectron> makePtrFromCollection(const std::vector<pat::Electron>& coll, unsigned idx) {
edm::Ptr<pat::Electron> temp(&coll,idx);
return edm::Ptr<reco::GsfElectron>(temp);
}

""")

ROOT.gROOT.ProcessLine("""
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

#include <iostream>

bool callGsfElectronVIDProducer(VersionedGsfElectronSelector& select, const edm::Ptr<reco::GsfElectron>& ele, const edm::EventBase& event) {
  return select(ele,event);
}

bool callGsfElectronVIDProducer(VersionedGsfElectronSelector& select, const edm::Ptr<reco::GsfElectron>& ele) {
  return select(ele);
}
""")

ROOT.gROOT.ProcessLine("""
#include <sstream>
#include <string>

std::string printVIDToString(const VersionedGsfElectronSelector& select) {
   std::stringstream out;
   select.print(out);
   return out.str();
}
""")

def noodle( collection, idx ):
    """ create a transient edm::Ptr to an object in a collection """
    the_ptr = ROOT.makePtrFromCollection(collection,idx)
    return the_ptr

def process_pset( pythonpset ):  
    """ turn a python cms.PSet into a VID ID """
    escaped_pset = config_template%(pythonpset)
    
    idname = str(pythonpset.idName).replace('-','_')
    idname = idname.replace("cms.string('","")
    idname = idname.replace("')","")
        
    return MakeGsfVersionedSelector()(escaped_pset,idname)

class VIDElectronSelector:
    def __init__(self,pythonpset = None):        
        self.initialized_ = False
        self.suffix_ = id_generator(7)
        self.instance_ = None
        if pythonpset is not None:
            self.instance_ = process_pset(pythonpset) 
            self.initialized_ = True
        else:
            self.instance_ = VersionedGsfElectronSelector()
    
    def __call__(self,*args):
        if( len(args) < 2 ):
            print 'call takes the following args: (the collection, index, <optional> event)'
            raise 
        temp = noodle(args[0],args[1])
        newargs = [temp] 
        if( len(args) == 3 ):
            newargs += [args[2].object().event()]
        return ROOT.callGsfElectronVIDProducer(self.instance_, *newargs)
        
    def initialize(self,pythonpset):
        if( self.initialized_ ): return
        config = process_pset(pythonpset)         
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

    def __repr__(self):
        return ROOT.printVIDToString(self.instance_)
        

    
