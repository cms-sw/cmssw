from FWCore.GuiBrowsers.ConfigToolBase import *


class AddTCMet(ConfigToolBase):
    """
    Tool to add track corrected MET collection to you PAT Tuple
    """
    _label='addTCMET'    
    _defaultParameters=dicttypes.SortedKeysDict()
    
    def __init__(self):
        """
        Initialize elements of the class. Note that the tool needs to be derived from ConfigToolBase
        to be usable in the configEditor.
        """
        ## initialization of the base class
        ConfigToolBase.__init__(self)
        ## add all parameters that should be known to the class
        self.addParameter(self._defaultParameters,'label_name','patTCMet', 'Label name of the new module.')
        ## set defaults 
        self._parameters=copy.deepcopy(self._defaultParameters)
        ## add comments
        self._comment = 'Add track corrected Met as PAT object to your PAt Tuple'

    def getDefaultParameters(self):
        """
        Return default parameters of the class
        """
        return self._defaultParameters

    def __call__(self,process,postfixLabel=None) :
        """
        Function call wrapper. This will check the parameters and call the actual implementation that
        can be found in toolCode.
        """
        if  lable_name is None:
            label_name = self._defaultParameters['label_name'].value 
        self.setParameter('label_name', label_name)
        self.apply(process) 
        
    def toolCode(self, process):                
        from PhysicsTools.PatAlgos.producersLayer1.metProducer_cfi import patMETs
        setattr(process, new_label, patMETs.clone(metSource = "tcMet"))
        process.patCandidateSummary.candidates += [ cms.InputTag(new_label) ]
       
addTCMet=AddTCMet()

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
