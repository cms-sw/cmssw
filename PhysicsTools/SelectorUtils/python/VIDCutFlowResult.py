import ROOT
import string
import random

# load FWLite C++ libraries
ROOT.gSystem.Load("libFWCoreFWLite.so");
ROOT.gSystem.Load("libDataFormatsFWLite.so");
ROOT.AutoLibraryLoader.enable()

class VIDCutFlowResult:
    def __init__(self, instance):
        self.__instance = instance

    def cutFlowName(self):
        return self.__instance.cutFlowName()
    
    def cutFlowPassed(self):
        return self.__instance.cutFlowPassed()

    def cutFlowSize(self):
        return self.__instance.cutFlowSize()

    def getNameAtIndex(self,idx):
        return self.__instance.getNameAtIndex(idx)

    def getCutResultByIndex(self,idx):
        return self.__instance.getCutResultByIndex(idx)

    def getCutResultByName(self,name):
        return self.__instance.getCutResultByName(name)

    def isCutMasked(self,idx_or_name):
        return self.__instance.isCutMasked(idx_or_name)

    def getValueCutUpon(self,idx_or_name):
        return self.__instance.getValueCutUpon(idx_or_name)
    
    def getCutFlowResultMasking(self,things_to_mask):       
        if type(things_to_mask) == str or type(things_to_mask) == int:
            return VIDCutFlowResult(self.__instance.getCutFlowResultMasking(things_to_mask))
        elif type(things_to_mask) != list:
            raise Exception('InvalidType','getCutFlowResultMasking only accepts (lists of) strings or ints!')
            
        if type(things_to_mask) == list:
            vect = None
            if len(things_to_mask) <= 0: 
                raise Exception('NothingToMask')
            if type(things_to_mask[0]) == str:
                vect = ROOT.std.vector('std::string')()
            elif type(things_to_mask[0]) == int:
                vect = ROOT.std.vector('unsigned int')()
            else:
                raise Exception('InvalidType','getCutFlowResultMasking only accepts (lists of) strings or ints!')
        
        for item in things_to_mask:
            vect.push_back(item)

        result = VIDCutFlowResult(self.__instance.getCutFlowResultMasking(vect))
        del vect
        
        return result

            
             
