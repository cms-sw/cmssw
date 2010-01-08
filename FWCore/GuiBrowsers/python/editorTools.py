import copy
import FWCore.ParameterSet.Config as cms
from FWCore.GuiBrowsers.ConfigToolBase import ConfigToolBase
from FWCore.ParameterSet.Types  import InputTag    

class UserCodeTool(ConfigToolBase):
    """ User code tool """
    _label="User code"
    _defaultParameters={}
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'code','', 'User code modifying the process: e.g. process.maxevents=1')
        self._parameters=copy.deepcopy(self._defaultParameters)  
    def dumpPython(self):
        dumpPython=""
        if self._comment!="":
            dumpPython = "#"+self._comment+"\n"
        dumpPython+=self._parameters['code'].value
        return ("",dumpPython)
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

class ChangeSource(ConfigToolBase):

    """ Tool for changing the source of a Process;
        Implemented for testing purposes.
    """
    
    _label='ChangeSource'
    
    _defaultParameters={}
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'source','No default value. Set your own', ' Source')
        self._parameters=copy.deepcopy(self._defaultParameters)
        
    def getDefaultParameters(self):
        return self._defaultParameters
   
    def dumpPython(self):
        dumpPythonImport = "\nfrom PhysicsTools.PatAlgos.tools.testTools import *\n"
        dumpPython=""
        if self._comment!="":
            dumpPython = "#"+self._comment
        dumpPython += "\nchangeSource(process, "
        dumpPython +='"'+ str(self.getvalue('source'))+'"'+')'+'\n'
        return (dumpPythonImport,dumpPython)

    def __call__(self,process,source=None) :
        if source is None:
           source=self._defaultParameters['source'].value 
        self.setParameter('source',source)
        self.apply(process) 
        
    def apply(self, process):
        source=self._parameters['source'].value
        process.disableRecording()
        process.source.fileNames=cms.untracked.vstring(source)
        process.enableRecording()
        action=self.__copy__()
        process.addAction(action)
    
changeSource=ChangeSource()
