from FWCore.GuiBrowsers.ConfigToolBase import *

from RecoHI.HiEgammaAlgos.HiHelperTools import *

class RestrictInputToAOD(ConfigToolBase):

    """ Remove pat object production steps which rely on RECO event
    content
    """
    _label='restrictInputToAOD'
    _defaultParameters=dicttypes.SortedKeysDict()
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'names',['All'], "list of collection names; supported are 'Photons', 'Electrons',, 'Muons', 'Taus', 'Jets', 'METs', 'All'", allowedValues=['Photons','Electrons', 'Muons', 'Taus', 'Jets', 'METs', 'All'])
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters
  
    def __call__(self,process,
                 names     = None) :
        if  names is None:
            names=self._defaultParameters['names'].value
        self.setParameter('names',names)
        self.apply(process) 
        
    def toolCode(self, process):        
        names=self._parameters['names'].value
        for obj in range(len(names)):
            print "---------------------------------------------------------------------"
            print "WARNING: the following additional information can only be used on "
            print "         RECO format:"
            if( names[obj] == 'Photons' or names[obj] == 'All' ):
                print "          * nothing needs to be done for Photons"
            if( names[obj] == 'Electrons' or names[obj] == 'All' ):
                print "          * nothing needs to be done for Electrons"            
            if( names[obj] == 'Muons' or names[obj] == 'All' ):
                print "          * nothing needs to be done for Muons"            
            if( names[obj] == 'Taus' or names[obj] == 'All' ):
                print "          * nothing needs to be done for Taus"            
            if( names[obj] == 'Jets' or names[obj] == 'All' ):
                print "          * nothing needs to be done for Jets"            
            if( names[obj] == 'METs' or names[obj] == 'All' ):
                print "          * nothing needs to be done for METs"            
        print "---------------------------------------------------------------------"
       
restrictInputToAOD=RestrictInputToAOD()


class RemoveMCMatching(ConfigToolBase):

    """ Remove monte carlo matching from a given collection or all PAT
    candidate collections:
    """
    _label='removeMCMatching'
    _defaultParameters=dicttypes.SortedKeysDict()
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'names',['All'], "collection name; supported are 'Photons','Muons', 'Taus', 'Jets', 'METs', 'All', 'PFAll','PFTaus','PFMuons'", allowedValues=['Photons','Muons', 'Taus', 'Jets', 'METs', 'All', 'PFAll','PFTaus','PFMuons'])
        self.addParameter(self._defaultParameters,'postfix',"", "postfix of default sequence")
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process,
                 names     = None,
                 postfix   = None) :
        if  names is None:
            names=self._defaultParameters['names'].value
        if postfix  is None:
            postfix=self._defaultParameters['postfix'].value
        self.setParameter('names',names)
        self.setParameter('postfix',postfix)
        self.apply(process) 
        
    def toolCode(self, process):        
        names=self._parameters['names'].value
        postfix=self._parameters['postfix'].value

        print "************** MC dependence removal ************"
        for obj in range(len(names)):    
            if( names[obj] == 'Photons'   or names[obj] == 'All' ):
                print "removing MC dependencies for photons"
                _removeMCMatchingForPATObject(process, 'photonMatch', 'patPhotons', postfix) 
            if( names[obj] == 'Muons'     or names[obj] == 'All' ):
                print "removing MC dependencies for muons"
                _removeMCMatchingForPATObject(process, 'muonMatch', 'patMuons', postfix) 
            if( names[obj] == 'Taus'      or names[obj] == 'All' ):
                print "removing MC dependencies for taus"
                _removeMCMatchingForPATObject(process, 'tauMatch', 'patTaus', postfix)
                ## remove mc extra modules for taus
                getattr(process,"patHeavyIonDefaultSequence"+postfix).remove(
                    applyPostfix(process, "tauGenJets", postfix))
                getattr(process,"patHeavyIonDefaultSequence"+postfix).remove(
                    applyPostfix(process, "tauGenJetsSelectorAllHadrons", postfix))
                getattr(process,"patHeavyIonDefaultSequence"+postfix).remove(
                    applyPostfix(process, "tauGenJetMatch", postfix))
                ## remove mc extra configs for taus
                tauProducer = getattr(process, 'patTaus'+postfix)
                tauProducer.addGenJetMatch      = False
                tauProducer.embedGenJetMatch    = False
                tauProducer.genJetMatch         = ''         
            if( names[obj] == 'Jets'  ):#    or names[obj] == 'All' ):
                print "removing MC dependencies for jets"
                ## remove mc extra modules for jets
                getattr(process,"patHeavyIonDefaultSequence"+postfix).remove(
                    applyPostfix(process, "patJetPartonMatch", postfix))
                getattr(process,"patHeavyIonDefaultSequence"+postfix).remove(
                    applyPostfix(process, "patJetGenJetMatch", postfix))
#                getattr(process,"patHeavyIonDefaultSequence"+postfix).remove(
#                    applyPostfix(process, "patJetFlavourId", postfix))
                ## remove mc extra configs for jets
                jetProducer = getattr(process, jetCollectionString()+postfix)
                jetProducer.addGenPartonMatch   = False
                jetProducer.embedGenPartonMatch = False
                jetProducer.genPartonMatch      = ''
                jetProducer.addGenJetMatch      = False
                jetProducer.genJetMatch         = ''
                jetProducer.getJetMCFlavour     = False
                jetProducer.JetPartonMapSource  = ''
                ## adjust output
                #         process.out.outputCommands.append("drop *_selectedPatJets*_genJets_*")
                
            if( names[obj] == 'METs'      or names[obj] == 'All' ):
                ## remove mc extra configs for jets
                metProducer = getattr(process, 'patMETs'+postfix)        
                metProducer.addGenMET           = False
                metProducer.genMETSource        = ''
            
removeMCMatching=RemoveMCMatching()

def _removeMCMatchingForPATObject(process, matcherName, producerName, postfix=""):
    ## remove mcMatcher from the default sequence
    objectMatcher = getattr(process, matcherName+postfix)
    if (producerName=='pfPatMuons'or producerName=='pfPatTaus'):
        #no idea what this should do: there is no other occurance of 'PFPATafterPAT' in CMSSW other than here...
        getattr(process,"PFPATafterPAT"+postfix).remove(objectMatcher)
    if (producerName=='patMuons'or producerName=='patTaus'or
        producerName=='patPhotons' or producerName=='patElectrons'):
        getattr(process,"patHeavyIonDefaultSequence"+postfix).remove(objectMatcher)
    ## straighten photonProducer
    objectProducer = getattr(process, producerName+postfix)
    objectProducer.addGenMatch      = False
    objectProducer.embedGenMatch    = False
    objectProducer.genParticleMatch = ''
    
    
class RemoveAllPATObjectsBut(ConfigToolBase):

    """ Remove all PAT objects from the default sequence but a specific one
    """
    _label='removeAllPATObjectsBut'
    _defaultParameters=dicttypes.SortedKeysDict()
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'names',self._defaultValue, "list of collection names; supported are 'Photons', 'Electrons', 'Muons', 'Taus', 'Jets', 'METs'", Type=list, allowedValues=['Photons', 'Electrons', 'Muons', 'Taus', 'Jets', 'METs'])
        self.addParameter(self._defaultParameters,'outputInProcess',True, "indicate whether there is an output module specified for the process (default is True)  ")
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process,
                 names               = None,
                 outputInProcess     = None) :
        if  names is None:
            names=self._defaultParameters['names'].value
        if  outputInProcess is None:
            outputInProcess=self._defaultParameters['outputInProcess'].value
        self.setParameter('names',names)
        self.setParameter('outputInProcess',outputInProcess)
        self.apply(process) 
        
    def toolCode(self, process):        
        names=self._parameters['names'].value
        outputInProcess=self._parameters['outputInProcess'].value

        removeTheseObjectCollections = ['Photons', 'Electrons', 'Muons', 'Taus', 'Jets', 'METs']
        for obj in range(len(names)):
            removeTheseObjectCollections.remove(names[obj])
        removeSpecificPATObjects(process, removeTheseObjectCollections, outputInProcess)
       
removeAllPATObjectsBut=RemoveAllPATObjectsBut()


class RemoveSpecificPATObjects(ConfigToolBase):

    """ Remove a specific PAT object from the default sequence
    """
    _label='removeSpecificPATObjects'
    _defaultParameters=dicttypes.SortedKeysDict()
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'names',self._defaultValue, "list of collection names; supported are 'Photons', 'Electrons', 'Muons', 'Taus', 'Jets', 'METs'", Type=list, allowedValues=['Photons', 'Electrons', 'Muons', 'Taus', 'Jets', 'METs'])
        self.addParameter(self._defaultParameters,'outputInProcess',True,"indicate whether there is an output module specified for the process (default is True)" )
        self.addParameter(self._defaultParameters,'postfix',"", "postfix of default sequence")
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process,
                 names               = None,
                 outputInProcess     = None,
                 postfix             = None) :
        if  names is None:
            names=self._defaultParameters['names'].value
        if  outputInProcess is None:
            outputInProcess=self._defaultParameters['outputInProcess'].value
        if postfix  is None:
            postfix=self._defaultParameters['postfix'].value
        self.setParameter('names',names)
        self.setParameter('outputInProcess',outputInProcess)
        self.setParameter('postfix',postfix)
        self.apply(process) 

    def toolCode(self, process):        
        names=self._parameters['names'].value
        outputInProcess=self._parameters['outputInProcess'].value
        postfix=self._parameters['postfix'].value
        ## remove pre object production steps from the default sequence

        for obj in range(len(names)):
            if( names[obj] == 'Photons' ):
                removeIfInSequence(process, 'patPhotonIsolation', "patHeavyIonDefaultSequence", postfix)
                removeIfInSequence(process, 'photonMatch', "patHeavyIonDefaultSequence", postfix)
            if( names[obj] == 'Muons' ):
                removeIfInSequence(process, 'muonMatch', "patHeavyIonDefaultSequence", postfix)
            if( names[obj] == 'Taus' ):
                removeIfInSequence(process, 'patPFCandidateIsoDepositSelection', "patHeavyIonDefaultSequence", postfix)
                removeIfInSequence(process, 'patPFTauIsolation', "patHeavyIonDefaultSequence", postfix)
                removeIfInSequence(process, 'tauMatch', "patHeavyIonDefaultSequence", postfix)
                removeIfInSequence(process, 'tauGenJets', "patHeavyIonDefaultSequence", postfix)
                removeIfInSequence(process, 'tauGenJetsSelectorAllHadrons', "patHeavyIonDefaultSequence", postfix)
                removeIfInSequence(process, 'tauGenJetMatch', "patHeavyIonDefaultSequence", postfix)
            if( names[obj] == 'Jets' ):
                removeIfInSequence(process, 'patJetCharge', "patHeavyIonDefaultSequence", postfix)
                removeIfInSequence(process, 'patJetCorrections', "patHeavyIonDefaultSequence", postfix)
                removeIfInSequence(process, 'patJetPartonMatch', "patHeavyIonDefaultSequence", postfix)
                removeIfInSequence(process, 'patJetGenJetMatch', "patHeavyIonDefaultSequence", postfix)
#                removeIfInSequence(process, 'patJetFlavourId', "patHeavyIonDefaultSequence", postfix)
            if( names[obj] == 'METs' ):
                removeIfInSequence(process, 'patMETCorrections', "patHeavyIonDefaultSequence", postfix)
        
            ## remove object production steps from the default sequence    
            if( names[obj] == 'METs' ):
                process.patCandidates.remove( getattr(process, 'pat'+names[obj]) )
            else:
                if( names[obj] == 'Jets' ):
                    applyPostfix(process,"patCandidates",postfix).remove(
                        getattr(process, jetCollectionString()+postfix) )
                    applyPostfix(process,"selectedPatCandidates",postfix).remove(
                        getattr(process, jetCollectionString('selected')+postfix) )
                    applyPostfix(process,"countPatCandidates",postfix).remove(
                        getattr(process, jetCollectionString('count')+postfix) )
                else:
                    applyPostfix(process,"patCandidates",postfix).remove( 
                        getattr(process, 'pat'+names[obj]+postfix) )
                    applyPostfix(process,"selectedPatCandidates",postfix).remove( 
                        getattr(process, 'selectedPat'+names[obj]+postfix) )
                    applyPostfix(process,"countPatCandidates",postfix).remove( 
                        getattr(process, 'countPat'+names[obj]+postfix) )
            ## in the case of leptons, the lepton counter must be modified as well
            if( names[obj] == 'Electrons' ):
                print 'removed from lepton counter: electrons'
                applyPostfix(process,"countPatLeptons",postfix).countElectrons = False
            elif( names[obj] == 'Muons' ):
                print 'removed from lepton counter: muons'
                applyPostfix(process,"countPatLeptons",postfix).countMuons = False
            elif( names[obj] == 'Taus' ):
                print 'removed from lepton counter: taus'
                applyPostfix(process,"countPatLeptons",postfix).countTaus = False
            ## remove from summary
            if( names[obj] == 'METs' ):
                applyPostfix(process,"patCandidateSummary",postfix).candidates.remove(
                    cms.InputTag('pat'+names[obj]+postfix) )
            else:
                if( names[obj] == 'Jets' ):
                    applyPostfix(process,"patCandidateSummary",postfix).candidates.remove( 
                        cms.InputTag(jetCollectionString()+postfix) )
                    applyPostfix(process,"selectedPatCandidateSummary",postfix).candidates.remove( 
                        cms.InputTag(jetCollectionString('selected')+postfix) )
                    applyPostfix(process,"cleanPatCandidateSummary",postfix).candidates.remove( 
                        cms.InputTag(jetCollectionString('clean')+postfix) )
                else:
                    applyPostfix(process,"patCandidateSummary",postfix).candidates.remove(
                        cms.InputTag('pat'+names[obj]+postfix) )
                    applyPostfix(process,"selectedPatCandidateSummary",postfix).candidates.remove( 
                        cms.InputTag('selectedPat'+names[obj]+postfix) )
                    getattr(process,"cleanPatCandidateSummary"+postfix).candidates.remove(
                        cms.InputTag('cleanPat'+names[obj]+postfix) )
        ## remove cleaning for the moment; in principle only the removed object
        ## could be taken out of the checkOverlaps PSet
        if ( outputInProcess ):
            print "---------------------------------------------------------------------"
            print "INFO   : some objects have been removed from the sequence. Switching "
            print "         off PAT cross collection cleaning, as it might be of limited"
            print "         sense now. If you still want to keep object collection cross"
            print "         cleaning within PAT you need to run and configure it by hand"
            removeCleaning(process)
    
               
removeSpecificPATObjects=RemoveSpecificPATObjects()


class RemoveCleaning(ConfigToolBase):

    """ remove PAT cleaning from the default sequence:
    """
    _label='removeCleaning'
    _defaultParameters=dicttypes.SortedKeysDict()
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'outputInProcess',True,"indicate whether there is an output module specified for the process (default is True)" )
        self.addParameter(self._defaultParameters,'postfix',"", "postfix of default sequence")
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process,
                 outputInProcess = None,
                 postfix         = None) :
        if  outputInProcess is None:
            outputInProcess=self._defaultParameters['outputInProcess'].value
        if postfix  is None:
            postfix=self._defaultParameters['postfix'].value

        self.setParameter('outputInProcess',outputInProcess)
        self.setParameter('postfix',postfix)

        self.apply(process) 
        
    def toolCode(self, process):        
        outputInProcess=self._parameters['outputInProcess'].value
        postfix=self._parameters['postfix'].value

        ## adapt single object counters
        for m in listModules(applyPostfix(process,"countPatCandidates",postfix)):
            if hasattr(m, 'src'): m.src = m.src.value().replace('cleanPat','selectedPat')

        ## adapt lepton counter
        countLept = applyPostfix(process,"countPatLeptons",postfix)
        countLept.electronSource = countLept.electronSource.value().replace('cleanPat','selectedPat')
        countLept.muonSource = countLept.muonSource.value().replace('cleanPat','selectedPat')
        countLept.tauSource = countLept.tauSource.value().replace('cleanPat','selectedPat')
        getattr(process, "patHeavyIonDefaultSequence"+postfix).remove(
            applyPostfix(process,"cleanPatCandidates",postfix)
            )
        if ( outputInProcess ):
            print "---------------------------------------------------------------------"
            print "INFO   : cleaning has been removed. Switch output from clean PAT     "
            print "         candidates to selected PAT candidates."
            ## add selected layer1 objects to the pat output
            from PhysicsTools.PatAlgos.patEventContent_cff import patEventContentNoCleaning
            process.out.outputCommands = patEventContentNoCleaning

           
removeCleaning=RemoveCleaning()


class AddCleaning(ConfigToolBase):

    """ Add PAT cleaning from the default sequence
    """
    _label='addCleaning'
    _defaultParameters=dicttypes.SortedKeysDict()
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'outputInProcess',True, "")
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process,
                 outputInProcess     = None):
        if  outputInProcess is None:
            outputInProcess=self._defaultParameters['outputInProcess'].value
        
        self.setParameter('outputInProcess',outputInProcess)
        self.apply(process) 
        
    def toolCode(self, process):        
        outputInProcess=self._parameters['outputInProcess'].value

        ## adapt single object counters
        process.patHeavyIonDefaultSequence.replace(process.countPatCandidates, process.cleanPatCandidates * process.countPatCandidates)
        for m in listModules(process.countPatCandidates):
            if hasattr(m, 'src'): m.src = m.src.value().replace('selectedPat','cleanPat')
        ## adapt lepton counter
        countLept = process.countPatLeptons
        countLept.electronSource = countLept.electronSource.value().replace('selectedPat','cleanPat')
        countLept.muonSource = countLept.muonSource.value().replace('selectedPat','cleanPat')
        countLept.tauSource = countLept.tauSource.value().replace('selectedPat','cleanPat')
        if ( outputInProcess ):
            print "---------------------------------------------------------------------"
            print "INFO   : cleaning has been added. Switch output from selected PAT    "
            print "         candidates to clean PAT candidates."
            ## add clean layer1 objects to the pat output
            from PhysicsTools.PatAlgos.patEventContent_cff import patEventContent
            process.out.outputCommands = patEventContent               
       
addCleaning=AddCleaning()
