import copy
import FWCore.ParameterSet.Config as cms
from FWCore.GuiBrowsers.ConfigToolBase import *
from FWCore.ParameterSet.Types  import InputTag    

class UserCodeTool(ConfigToolBase):
    """ User code tool """
    _label="userCode"
    _defaultParameters=dicttypes.SortedKeysDict()
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'code','', 'User code modifying the process: e.g. process.maxevents=1')
        self._parameters=copy.deepcopy(self._defaultParameters)  
        self._comment = ""
    def dumpPython(self):
        dumpPython=""
        if self._comment!="":
            dumpPython = "#"+self._comment+"\n"
        dumpPython+=self._parameters['code'].value
        return ("",dumpPython)
    def __call__(self,process,code):
        self.setParameter('code',code)
        self.apply(process)
        return self
    def toolCode(self,process):
        code=self._parameters['code'].value
        exec(code)

userCodeTool=UserCodeTool()

class ChangeSource(ConfigToolBase):

    """ Tool for changing the source of a Process;
        Implemented for testing purposes.
    """
    
    _label='changeSource'
    _defaultParameters=dicttypes.SortedKeysDict()
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'source','No default value. Set your own', ' Source filenames')
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""
        
    def getDefaultParameters(self):
        return self._defaultParameters
   
    def __call__(self,process,source=None) :
        if source is None:
           source=self._defaultParameters['source'].value 
        self.setParameter('source',source)
        self.apply(process) 
        
    def toolCode(self, process):
        source=self._parameters['source'].value
        process.source.fileNames=cms.untracked.vstring(source)
    
changeSource=ChangeSource()

from FWCore.ParameterSet.Modules import Source

if __name__=='__main__':
    import unittest
    class TestEditorTools(unittest.TestCase):
        def setUp(self):
            pass
        def testdumpPython(self):
            process = cms.Process('unittest')
            process.source=Source("PoolSource",fileNames = cms.untracked.string("file:file.root"))
            
            changeSource(process,"file:filename.root")
            self.assertEqual(changeSource.dumpPython(),  ('\nfrom FWCore.GuiBrowsers.editorTools import *\n', "\nchangeSource(process , 'file:filename.root')\n"))
            
    unittest.main()
