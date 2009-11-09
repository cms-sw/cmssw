import copy
import FWCore.ParameterSet.Config as cms
from FWCore.GuiBrowsers.ConfigToolBase import ConfigToolBase

class UserCodeTool(ConfigToolBase):
    """ User code tool """
    _label="User code"
    _defaultParameters={}
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'code','', 'User code modifying the process: e.g. process.maxevents=1')
        self._parameters=copy.deepcopy(self._defaultParameters)  
    def dumpPython(self):
        return ("",self._parameters['code'].value)
    def __call__(self,process,code):
        self.setParameter(self._parameters,'code',code, 'User code modifying the process: e.g. process.maxevents=1')
        self.apply(process)
        return self
    def apply(self,process):
        code=self._parameters['code'].value
        exec code
    def typeError(self,name,bool):
        pass

userCodeTool=UserCodeTool()
