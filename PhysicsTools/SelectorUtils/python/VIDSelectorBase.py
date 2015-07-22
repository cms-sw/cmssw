import ROOT
import string
import random
import sys

from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry
from PhysicsTools.SelectorUtils.VIDCutFlowResult import VIDCutFlowResult
import DataFormats.FWLite

# load FWLite C++ libraries
ROOT.gSystem.Load("libFWCoreFWLite.so");
ROOT.gSystem.Load("libDataFormatsFWLite.so");
ROOT.FWLiteEnabler.enable()

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
            if hasattr(pythonpset,'isPOGApproved'):
                approved = pythonpset.isPOGApproved.value()
                if not approved:
                    sys.stderr.write('This ID is not POG approved and likely under development!!!!\n')
                    sys.stderr.write('Please make sure to report your progress with this ID'\
                                         ' at the next relevant POG meeting.\n')
                del pythonpset.isPOGApproved
            else:
                sys.stderr.write('This ID is not POG approved and likely under development!!!!\n')
                sys.stderr.write('Please make sure to report your progress with this ID'\
                                     ' at the next relevant POG meeting.\n')
            self.__instance = process_pset( self.__selectorBuilder, pythonpset ) 
            expectedmd5 = central_id_registry.getMD5FromName(pythonpset.idName)
            if expectedmd5 != self.md5String():
                sys.stderr.write("ID: %s\n"%self.name())
                sys.stderr.write("The expected md5: %s does not match the md5\n"%expectedmd5)
                sys.stderr.write("calculated by the ID: %s please\n"%self.md5String())
                sys.stderr.write("update your python configuration or determine the source\n")
                sys.stderr.write("of transcription error!\n")
            self.__initialized = True
        else:
            self.__instance = self.__selectorBuilder()
    
    def __call__(self,*args):
        if( len(args) == 1 ):
            return self.__instance(*args)
        if( len(args) == 2 and isinstance(args[1],DataFormats.FWLite.Events) ):
            return self.__instance(args[0],args[1].object().event())
        elif( len(args) == 2 and type(args[1]) is int ):
            temp = self.__ptrMaker(args[0],args[1])
            newargs = [temp] 
            return self.__instance(*newargs)
        if( len(args) == 3 ):
            temp = self.__ptrMaker(args[0],args[1])
            newargs = [temp]
            newargs += [args[2].object().event()]
            return self.__instance(*newargs)
        
    def initialize(self,pythonpset):
        if( self.__initialized ): 
            print 'VID Selector is already initialized, doing nothing!'
            return
        del process.__instance
        if hasattr(pythonpset,'isPOGApproved'):
            approved = pythonpset.isPOGApproved.value()
            if not approved:
                sys.stderr.write('This ID is not POG approved and likely under development!!!!\n')
                sys.stderr.write('Please make sure to report your progress with this ID'\
                                     ' at the next relevant POG meeting.\n')
            del pythonpset.isPOGApproved
        else:
            sys.stderr.write('This ID is not POG approved and likely under development!!!!\n')
            sys.stderr.write('Please make sure to report your progress with this ID'\
                                 ' at the next relevant POG meeting.\n')
        self.__instance = process_pset( self.__selectorBuilder, pythonpset )         
        expectedmd5 = central_id_registry.getMD5FromName(pythonpset.idName)
        if expectedmd5 != self.md5String():
            sys.stderr.write("ID: %s\n"%self.name())
            sys.stderr.write("The expected md5: %s does not match the md5\n"%expectedmd5)
            sys.stderr.write("calculated by the ID: %s please\n"%self.md5String())
            sys.stderr.write("update your python configuration or determine the source\n")
            sys.stderr.write("of transcription error!\n")
        self.__initialized = True

    def cutFlowSize(self):
        return self.__instance.cutFlowSize()

    def cutFlowResult(self):
        return VIDCutFlowResult(self.__instance.cutFlowResult())

    def howFarInCutFlow(self):
        return self.__instance.howFarInCutFlow()

    def name(self):
        return self.__instance.name()

    def bitMap(self):
        return self.__instance.bitMap()

    def md5String(self):
        return self.__instance.md5String()

    def md55Raw(self):
        return self.__instance.md55Raw()

    def __repr__(self):
        return self.__printer(self.__instance)
