from FWCore.GuiBrowsers.ConfigToolBase import *


class AddTcMET(ConfigToolBase):

    """ Add track corrected MET collection to patEventContent
    """
    _label='addTcMET'    
    _defaultParameters=dicttypes.SortedKeysDict()
    
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'postfixLabel','TC', '')
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ''

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process,postfixLabel=None) :
        if  postfixLabel is None:
            postfixLabel=self._defaultParameters['postfixLabel'].value 
        self.setParameter('postfixLabel',postfixLabel)
        self.apply(process) 
        
    def toolCode(self, process):                
        postfixLabel=self._parameters['postfixLabel'].value


        ## add module as process to the default sequence
        def addAlso (label,value):
            existing = getattr(process, label)
            setattr( process, label+postfixLabel, value)
            process.patDefaultSequence.replace( existing, existing*value )        

        ## clone and add a module as process to the
        ## default sequence
        def addClone(label,**replaceStatements):
            new = getattr(process, label).clone(**replaceStatements)
            addAlso(label, new)

        ## addClone('corMetType1Icone5Muons', uncorMETInputTag = cms.InputTag("tcMet"))
        addClone('patMETs', metSource = cms.InputTag("tcMet"))

        ## add new met collections output to the pat summary
        process.patCandidateSummary.candidates += [ cms.InputTag('patMETs'+postfixLabel) ]
       

addTcMET=AddTcMET()

class AddPfMET(ConfigToolBase):
    
    """ Add pflow MET collection to patEventContent
    """
    _label='addPfMET'    
    _defaultParameters=dicttypes.SortedKeysDict()
    
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'postfixLabel','PF', '')
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ''
        
    def getDefaultParameters(self):
        return self._defaultParameters
        
    def __call__(self,process,postfixLabel=None):
        if  postfixLabel is None:
            postfixLabel=self._defaultParameters['postfixLabel'].value 
        self.setParameter('postfixLabel',postfixLabel)
        self.apply(process) 

    def toolCode(self, process): 
        postfixLabel=self._parameters['postfixLabel'].value


        ## add module as process to the default sequence
        def addAlso (label,value):
            existing = getattr(process, label)
            setattr( process, label+postfixLabel, value)
            process.patDefaultSequence.replace( existing, existing*value )        
            
        ## clone and add a module as process to the
        ## default sequence
        def addClone(label,**replaceStatements):
            new = getattr(process, label).clone(**replaceStatements)
            addAlso(label, new)

        ## addClone('corMetType1Icone5Muons', uncorMETInputTag = cms.InputTag("tcMet"))
        addClone('patMETs', metSource = cms.InputTag("pfType1CorrectedMet"), addMuonCorrections = False)

        ## add new met collections output to the pat summary
        process.patCandidateSummary.candidates += [ cms.InputTag('patMETs'+postfixLabel) ]

       
addPfMET=AddPfMET()
