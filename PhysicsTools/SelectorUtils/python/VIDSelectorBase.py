import ROOT
import string
import random

# load FWLite C++ libraries
ROOT.gSystem.Load("libFWCoreFWLite.so");
ROOT.gSystem.Load("libDataFormatsFWLite.so");
ROOT.AutoLibraryLoader.enable()

config_template = """
import FWCore.ParameterSet.Config as cms

from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

ebCutOff = 1.479

%s
"""

def process_pset( builder, pythonpset ):  
    """ turn a python cms.PSet into a VID ID """
    escaped_pset = config_template%(pythonpset)
    
    idname = pythonpset.idName.value().replace('-','_')
        
    return builder(escaped_pset,idname)

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

class VIDSelectorBase:
    def __init__(self, vidSelectorBuilder, ptrMaker, printer, pythonpset = None):        
        self.__initialized = False
        self.__suffix = id_generator(7)
        self.__printer = printer()
        self.__ptrMaker = ptrMaker()
        self.__selectorBuilder = vidSelectorBuilder()
        self.__instance = None
        if pythonpset is not None:
            self.__instance = process_pset( self.__selectorBuilder, pythonpset ) 
            self.__initialized = True
        else:
            self.__instance = self.__selectorBuilder()
    
    def __call__(self,*args):
        if( len(args) < 2 ):
            print 'call takes the following args: (the collection, index, <optional> event)'
            raise 
        temp = self.__ptrMaker(args[0],args[1])
        newargs = [temp] 
        if( len(args) == 3 ):
            newargs += [args[2].object().event()]
        return self.__instance(*newargs)
        
    def initialize(self,pythonpset):
        if( self.__initialized ): 
            print 'VID Selector is already initialized, doing nothing!'
            return
        del process.__instance
        self.__instance = process_pset( self.__selectorBuilder, pythonpset )         
        self.__initialized = True

    def cutFlowSize(self):
        return self.__instance.cutFlowSize()

    def howFarInCutFlow(self):
        return self.__instance.howFarInCutFlow()

    def name(self):
        return self.__instance.name()

    def md5String(self):
        return self.__instance.md5String()

    def md55Raw(self):
        return self.__instance.md55Raw()

    def __repr__(self):
        return self.__printer(self.__instance)
