from FWCore.GuiBrowsers.ConfigToolBase import *

from PhysicsTools.PatAlgos.patEventContent_cff import *


class SwitchOnTrigger(ConfigToolBase):    
    """ Enables trigger information in PAT
    """    
    _label='switchOnTrigger'    
    _defaultParameters=dicttypes.SortedKeysDict()
    
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'outputInProcess',True, "indicate whether there is an output module specified for the process (default is\True)  ")
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process,
                  outputInProcess     = None) :
        if  outputInProcess is None:
            outputInProcess=self._defaultParameters['outputInProcess'].value
        self.setParameter('outputInProcess',outputInProcess)
        self.apply(process) 
        
    def toolCode(self, process):
        outputInProcess=self._parameters['outputInProcess'].value
         
        ## add trigger modules to path
        process.load("PhysicsTools.PatAlgos.triggerLayer1.triggerProducer_cff")
        process.patDefaultSequence += process.patTriggerSequence
        ## configure pat trigger
        process.patTrigger.onlyStandAlone = False
        ## add trigger specific event content to PAT event content
        if ( outputInProcess ):
            process.out.outputCommands += patTriggerEventContent
            for matchLabel in process.patTriggerEvent.patTriggerMatches:
                process.out.outputCommands += [ 'keep patTriggerObjectsedmAssociation_patTriggerEvent_' + matchLabel + '_*' ]

switchOnTrigger=SwitchOnTrigger()

class SwitchOnTriggerStandAlone(ConfigToolBase):
    """
    """
    _label='switchOnTriggerStandAlone'    
    _defaultParameters=dicttypes.SortedKeysDict()
    
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'outputInProcess',True, "indicate whether there is an output module specified for the process (default is\True)  ")
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process,
                 outputInProcess     = None) :
        if  outputInProcess is None:
            outputInProcess=self._defaultParameters['outputInProcess'].value
        self.setParameter('outputInProcess',outputInProcess)
        self.apply(process) 
        
    def toolCode(self, process):
        outputInProcess=self._parameters['outputInProcess'].value
        ## add trigger modules to path
        process.load("PhysicsTools.PatAlgos.triggerLayer1.triggerProducer_cff")
        process.patDefaultSequence += process.patTriggerSequence
        ## configure pat trigger
        process.patTrigger.onlyStandAlone = True
        process.patTriggerSequence.remove( process.patTriggerEvent )
        if ( outputInProcess ):
            process.out.outputCommands += patTriggerStandAloneEventContent

      
        
switchOnTriggerStandAlone=SwitchOnTriggerStandAlone()

class SwitchOnTriggerAll(ConfigToolBase):
    """
    """
    _label='switchOnTriggerAll'    
    _defaultParameters=dicttypes.SortedKeysDict()
    
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'outputInProcess',True, "indicate whether there is an output module specified for the process (default is\True)  ")
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process,
                 outputInProcess     = None) :
        if  outputInProcess is None:
            outputInProcess=self._defaultParameters['outputInProcess'].value
        self.setParameter('outputInProcess',outputInProcess)
        self.apply(process) 
        
    def toolCode(self, process):
        outputInProcess=self._parameters['outputInProcess'].value
        switchOnTrigger( process )
        if ( outputInProcess ):
            process.out.outputCommands += patTriggerStandAloneEventContent
      

switchOnTriggerAll=SwitchOnTriggerAll()
        
class SwitchOnTriggerMatchEmbedding(ConfigToolBase):
    """
    """
    _label='switchOnTriggerMatchEmbedding'    
    _defaultParameters=dicttypes.SortedKeysDict()
    
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'outputInProcess',True, "indicate whether there is an output module specified for the process (default is\True)  ")
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process,
                 outputInProcess     = None) :
        if  outputInProcess is None:
            outputInProcess=self._defaultParameters['outputInProcess'].value
        self.setParameter('outputInProcess',outputInProcess)
        self.apply(process) 
        
    def toolCode(self, process):
        outputInProcess=self._parameters['outputInProcess'].value
        process.patTriggerSequence += process.patTriggerMatchEmbedder
        if ( outputInProcess ):
            process.out.outputCommands += patEventContentTriggerMatch
      

switchOnTriggerMatchEmbedding=SwitchOnTriggerMatchEmbedding()
