from FWCore.GuiBrowsers.ConfigToolBase import *

from PhysicsTools.PatAlgos.patEventContent_cff import *

path = "PhysicsTools.PatAlgos.tools.trigTools"

class SwitchOnTrigger(ConfigToolBase):    
    """ Enables trigger information in PAT
    """    
    _label='switchOnTrigger'    
    _defaultParameters=dicttypes.SortedKeysDict()
    _path = path
    def __init__(self):
        ConfigToolBase.__init__(self)
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process) :
        self.apply(process) 
        
    def toolCode(self, process):       
        ## add trigger modules to path
        process.load("PhysicsTools.PatAlgos.triggerLayer1.triggerProducer_cff")
        process.patDefaultSequence += process.patTriggerSequence
        ## configure pat trigger
        process.patTrigger.onlyStandAlone = False
        ## add trigger specific event content to PAT event content
        process.out.outputCommands += patTriggerEventContent
        for matchLabel in process.patTriggerEvent.patTriggerMatches:
            process.out.outputCommands += [ 'keep patTriggerObjectsedmAssociation_patTriggerEvent_' + matchLabel + '_*' ]

switchOnTrigger=SwitchOnTrigger()

class SwitchOnTriggerStandAlone(ConfigToolBase):
    """
    """
    _label='switchOnTriggerStandAlone'    
    _defaultParameters=dicttypes.SortedKeysDict()
    _path = path
    def __init__(self):
        ConfigToolBase.__init__(self)
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process) :
        self.apply(process) 
        
    def toolCode(self, process):
        ## add trigger modules to path
        process.load("PhysicsTools.PatAlgos.triggerLayer1.triggerProducer_cff")
        process.patDefaultSequence += process.patTriggerSequence
        ## configure pat trigger
        process.patTrigger.onlyStandAlone = True
        process.patTriggerSequence.remove( process.patTriggerEvent )
        process.out.outputCommands += patTriggerStandAloneEventContent

      
        
switchOnTriggerStandAlone=SwitchOnTriggerStandAlone()

class SwitchOnTriggerAll(ConfigToolBase):
    """
    """
    _label='switchOnTriggerAll'    
    _defaultParameters=dicttypes.SortedKeysDict()
    _path = path
    def __init__(self):
        ConfigToolBase.__init__(self)
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process) :
        self.apply(process) 
        
    def toolCode(self, process):
        switchOnTrigger( process )
        process.out.outputCommands += patTriggerStandAloneEventContent
      

switchOnTriggerAll=SwitchOnTriggerAll()
        
class SwitchOnTriggerMatchEmbedding(ConfigToolBase):
    """
    """
    _label='switchOnTriggerMatchEmbedding'    
    _defaultParameters=dicttypes.SortedKeysDict()
    _path = path
    def __init__(self):
        ConfigToolBase.__init__(self)
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process) :
        self.apply(process) 
        
    def toolCode(self, process):
        process.patTriggerSequence += process.patTriggerMatchEmbedder
        process.out.outputCommands += patEventContentTriggerMatch
      

switchOnTriggerMatchEmbedding=SwitchOnTriggerMatchEmbedding()
