from FWCore.GuiBrowsers.ConfigToolBase import *

from PhysicsTools.PatAlgos.tools.helpers import *

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


class RunOnData(ConfigToolBase):

    """ Remove monte carlo matching from a given collection or all PAT
    candidate collections and adapt the JEC's:
    """
    _label='runOnData'
    _defaultParameters=dicttypes.SortedKeysDict()
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'names',['All'], "collection name; supported are 'Photons', 'Electrons','Muons', 'Taus', 'Jets', 'METs', 'All', 'PFAll', 'PFElectrons','PFTaus','PFMuons'", allowedValues=['Photons', 'Electrons','Muons', 'Taus', 'Jets', 'METs', 'All', 'PFAll', 'PFElectrons','PFTaus','PFMuons'])
        self.addParameter(self._defaultParameters,'postfix',"", "postfix of default sequence")
        self.addParameter(self._defaultParameters,'outputModules',['out'], "names of all output modules specified to be adapted (default is ['out'])")
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process,
                 names           = None,
                 postfix         = None,
                 outputInProcess = None,
                 outputModules   = None) :
        ## stop processing if 'outputInProcess' exists and show the new alternative
        if  not outputInProcess is None:
            deprecatedOptionOutputInProcess(self)
        if  names is None:
            names=self._defaultParameters['names'].value
        if  postfix  is None:
            postfix=self._defaultParameters['postfix'].value
        if  outputModules is None:
            outputModules=self._defaultParameters['outputModules'].value
        self.setParameter('names',names)
        self.setParameter('postfix',postfix)
        self.setParameter('outputModules',outputModules)
        self.apply(process)

    def toolCode(self, process):
        names=self._parameters['names'].value
        postfix=self._parameters['postfix'].value
        outputModules=self._parameters['outputModules'].value

        print '******************* RunOnData *******************'
        removeMCMatching(process, names=names, postfix=postfix, outputModules=outputModules)
        for mod in getattr(process,'patDefaultSequence'+postfix).moduleNames():
            if mod.startswith('patJetCorrFactors'):
                prefix = getattr(process, mod).payload.pythonValue().replace("'","")
                if 'L3Absolute' in getattr(process,mod).levels:
                    if not 'L2L3Residual' in getattr(process,mod).levels:
                        getattr(process,mod).levels.insert(getattr(process,mod).levels.index('L3Absolute')+1, 'L2L3Residual')
                        print 'adding L2L3Residual JEC for:', getattr(process,mod).label_()
                if hasattr(process, prefix+'CombinedCorrector'+postfix):
                    if prefix+'L3Absolute' in getattr(process,prefix+'CombinedCorrector'+postfix).correctors:
                        if not prefix+'L2L3Residual' in getattr(process,prefix+'CombinedCorrector'+postfix).correctors:
                            idx = getattr(process,prefix+'CombinedCorrector'+postfix).correctors.index(prefix+'L3Absolute')+1
                            getattr(process,prefix+'CombinedCorrector'+postfix).correctors.insert(idx, prefix+'L2L3Residual')
                            print 'adding L2L3Residual for TypeI MET correction:', getattr(process,prefix+'CombinedCorrector'+postfix).label_()

runOnData=RunOnData()


class RemoveMCMatching(ConfigToolBase):

    """ Remove monte carlo matching from a given collection or all PAT
    candidate collections:
    """
    _label='removeMCMatching'
    _defaultParameters=dicttypes.SortedKeysDict()
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'names',['All'], "collection name; supported are 'Photons', 'Electrons','Muons', 'Taus', 'Jets', 'METs', 'All', 'PFAll', 'PFElectrons','PFTaus','PFMuons'", allowedValues=['Photons', 'Electrons','Muons', 'Taus', 'Jets', 'METs', 'All', 'PFAll', 'PFElectrons','PFTaus','PFMuons'])
        self.addParameter(self._defaultParameters,'postfix',"", "postfix of default sequence")
        self.addParameter(self._defaultParameters,'outputInProcess',True, "indicates whether the output of the pat tuple should be made persistent or not (legacy)")
        self.addParameter(self._defaultParameters,'outputModules',['out'], "names of all output modules specified to be adapted (default is ['out'])")
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process,
                 names           = None,
                 postfix         = None,
                 outputInProcess = None,
                 outputModules   = None) :
        ## stop processing if 'outputInProcess' exists and show the new alternative
        if  not outputInProcess is None:
            deprecatedOptionOutputInProcess(self)
        else:
            outputInProcess=self._parameters['outputInProcess'].value
        if  names is None:
            names=self._defaultParameters['names'].value
        if postfix  is None:
            postfix=self._defaultParameters['postfix'].value
        if  outputModules is None:
            outputModules=self._defaultParameters['outputModules'].value
        self.setParameter('names',names)
        self.setParameter('postfix',postfix)
        self.setParameter('outputInProcess', outputInProcess)
        self.setParameter('outputModules',outputModules)
        self.apply(process)

    def toolCode(self, process):
        names=self._parameters['names'].value
        postfix=self._parameters['postfix'].value
        outputInProcess=self._parameters['outputInProcess'].value
        outputModules=self._parameters['outputModules'].value

        if not outputInProcess:
            outputModules=['']
        
        print "************** MC dependence removal ************"
        for obj in range(len(names)):
            if( names[obj] == 'Photons'   or names[obj] == 'All' ):
                print "removing MC dependencies for photons"
                _removeMCMatchingForPATObject(process, 'photonMatch', 'patPhotons', postfix)
            if( names[obj] == 'Electrons' or names[obj] == 'All' ):
                print "removing MC dependencies for electrons"
                _removeMCMatchingForPATObject(process, 'electronMatch', 'patElectrons', postfix)
            if( names[obj] == 'Muons'     or names[obj] == 'All' ):
                print "removing MC dependencies for muons"
                _removeMCMatchingForPATObject(process, 'muonMatch', 'patMuons', postfix)
            if( names[obj] == 'Taus'      or names[obj] == 'All' ):
                print "removing MC dependencies for taus"
                _removeMCMatchingForPATObject(process, 'tauMatch', 'patTaus', postfix)
                ## remove mc extra modules for taus
                for mod in ['tauGenJets','tauGenJetsSelectorAllHadrons','tauGenJetMatch']:
                    if hasattr(process,mod+postfix):
                        getattr(process,'patDefaultSequence'+postfix).remove(getattr(process,mod+postfix))
                ## remove mc extra configs for taus
                tauProducer = getattr(process,'patTaus'+postfix)
                tauProducer.addGenJetMatch   = False
                tauProducer.embedGenJetMatch = False
                tauProducer.genJetMatch      = ''
            if( names[obj] == 'Jets'      or names[obj] == 'All' ):
                print "removing MC dependencies for jets"
                ## there may be multiple jet collection, therefore all jet collections
                ## in patDefaultSequence+postfix are threated here
                jetPostfixes = []
                for mod in getattr(process,'patDefaultSequence'+postfix).moduleNames():
                    if mod.startswith('patJets'):
                        jetPostfixes.append(getattr(process, mod).label_().replace("patJets",""))
                for pfix in jetPostfixes:
                    ## remove mc extra modules for jets
                    for mod in ['patJetPartonMatch','patJetGenJetMatch','patJetFlavourIdLegacy','patJetPartonsLegacy','patJetPartonAssociationLegacy','patJetFlavourAssociationLegacy','patJetFlavourId','patJetPartons','patJetFlavourAssociation']:
                        if hasattr(process,mod+pfix):
                            getattr(process,'patDefaultSequence'+postfix).remove(getattr(process,mod+pfix))
                    ## remove mc extra configs for jets
                    jetProducer = getattr(process, jetCollectionString()+pfix)
                    jetProducer.addGenPartonMatch   = False
                    jetProducer.embedGenPartonMatch = False
                    jetProducer.genPartonMatch      = ''
                    jetProducer.addGenJetMatch      = False
                    jetProducer.genJetMatch         = ''
                    jetProducer.getJetMCFlavour     = False
                    jetProducer.useLegacyJetMCFlavour = False
                    jetProducer.addJetFlavourInfo   = False
                    jetProducer.JetPartonMapSource  = ''
                    jetProducer.JetFlavourInfoSource = ''
                ## adjust output
                for outMod in outputModules:
                    if hasattr(process,outMod):
                        getattr(process,outMod).outputCommands.append("drop *_selectedPatJets*_genJets_*")
                    else:
                        raise KeyError, "process has no OutModule named", outMod

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
        getattr(process,"patDefaultSequence"+postfix).remove(objectMatcher)
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
        self.addParameter(self._defaultParameters,'outputModules',['out'], "names of all output modules specified to be adapted (default is ['out'])")
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process,
                 names           = None,
                 outputInProcess = None,
                 outputModules   = None) :
        ## stop processing if 'outputInProcess' exists and show the new alternative
        if  not outputInProcess is None:
            deprecatedOptionOutputInProcess(self)
        if  names is None:
            names=self._defaultParameters['names'].value
        if  outputModules is None:
            outputModules=self._defaultParameters['outputModules'].value
        self.setParameter('names',names)
        self.setParameter('outputModules',outputModules)
        self.apply(process)

    def toolCode(self, process):
        names=self._parameters['names'].value
        outputModules=self._parameters['outputModules'].value

        removeTheseObjectCollections = ['Photons', 'Electrons', 'Muons', 'Taus', 'Jets', 'METs']
        for obj in range(len(names)):
            removeTheseObjectCollections.remove(names[obj])
        removeSpecificPATObjects(process, removeTheseObjectCollections, outputModules = outputModules)

removeAllPATObjectsBut=RemoveAllPATObjectsBut()


class RemoveSpecificPATObjects(ConfigToolBase):

    """ Remove a specific PAT object from the default sequence
    """
    _label='removeSpecificPATObjects'
    _defaultParameters=dicttypes.SortedKeysDict()
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'names',self._defaultValue, "list of collection names; supported are 'Photons', 'Electrons', 'Muons', 'Taus', 'Jets', 'METs'", Type=list, allowedValues=['Photons', 'Electrons', 'Muons', 'Taus', 'Jets', 'METs'])
        self.addParameter(self._defaultParameters,'outputModules',['out'], "names of all output modules specified to be adapted (default is ['out'])")
        self.addParameter(self._defaultParameters,'postfix',"", "postfix of default sequence")
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process,
                 names           = None,
                 outputInProcess = None,
                 postfix         = None,
                 outputModules   = None) :
        ## stop processing if 'outputInProcess' exists and show the new alternative
        if  not outputInProcess is None:
            deprecatedOptionOutputInProcess(self)
        if  names is None:
            names=self._defaultParameters['names'].value
        if  outputModules is None:
            outputModules=self._defaultParameters['outputModules'].value
        if postfix  is None:
            postfix=self._defaultParameters['postfix'].value
        self.setParameter('names',names)
        self.setParameter('outputModules',outputModules)
        self.setParameter('postfix',postfix)
        self.apply(process)

    def toolCode(self, process):
        names=self._parameters['names'].value
        outputModules=self._parameters['outputModules'].value
        postfix=self._parameters['postfix'].value

        ## remove pre object production steps from the default sequence
        for obj in range(len(names)):
            if( names[obj] == 'Photons' ):
                removeIfInSequence(process, 'patPhotonIsolation', "patDefaultSequence", postfix)
                removeIfInSequence(process, 'photonMatch', "patDefaultSequence", postfix)
                removeIfInSequence(process, 'patPhotons', "patDefaultSequence", postfix)
            if( names[obj] == 'Electrons' ):
                removeIfInSequence(process, 'patElectronId', "patDefaultSequence", postfix)
                removeIfInSequence(process, 'patElectronIsolation', "patDefaultSequence", postfix)
                removeIfInSequence(process, 'electronMatch', "patDefaultSequence", postfix)
                removeIfInSequence(process, 'patElectrons', "patDefaultSequence", postfix)
            if( names[obj] == 'Muons' ):
                removeIfInSequence(process, 'muonMatch', "patDefaultSequence", postfix)
                removeIfInSequence(process, 'patMuons', "patDefaultSequence", postfix)
            if( names[obj] == 'Taus' ):
                removeIfInSequence(process, 'patPFCandidateIsoDepositSelection', "patDefaultSequence", postfix)
                removeIfInSequence(process, 'patPFTauIsolation', "patDefaultSequence", postfix)
                removeIfInSequence(process, 'tauMatch', "patDefaultSequence", postfix)
                removeIfInSequence(process, 'tauGenJets', "patDefaultSequence", postfix)
                removeIfInSequence(process, 'tauGenJetsSelectorAllHadrons', "patDefaultSequence", postfix)
                removeIfInSequence(process, 'tauGenJetMatch', "patDefaultSequence", postfix)
                removeIfInSequence(process, 'patTaus', "patDefaultSequence", postfix)
            if( names[obj] == 'Jets' ):
                removeIfInSequence(process, 'patJetCharge', "patDefaultSequence", postfix)
                removeIfInSequence(process, 'patJetCorrections', "patDefaultSequence", postfix)
                removeIfInSequence(process, 'patJetPartonMatch', "patDefaultSequence", postfix)
                removeIfInSequence(process, 'patJetGenJetMatch', "patDefaultSequence", postfix)
                removeIfInSequence(process, 'patJetFlavourIdLegacy', "patDefaultSequence", postfix)
                removeIfInSequence(process, 'patJetFlavourId', "patDefaultSequence", postfix)
            if( names[obj] == 'METs' ):
                removeIfInSequence(process, 'patMETCorrections', "patDefaultSequence", postfix)

            ## remove object production steps from the default sequence
            if( names[obj] == 'METs' ):
                process.patDefaultSequence.remove( getattr(process, 'pat'+names[obj]) )
            else:
                if( names[obj] == 'Jets' ):
                    applyPostfix(process,"patDefaultSequence",postfix).remove(
                        getattr(process, jetCollectionString()+postfix) )
                    applyPostfix(process,"patDefaultSequence",postfix).remove(
                        getattr(process, jetCollectionString('selected')+postfix) )
                    applyPostfix(process,"patDefaultSequence",postfix).remove(
                        getattr(process, jetCollectionString('count')+postfix) )
                else:
                    applyPostfix(process,"patDefaultSequence",postfix).remove(
                        getattr(process, 'pat'+names[obj]+postfix) )
                    applyPostfix(process,"patDefaultSequence",postfix).remove(
                        getattr(process, 'selectedPat'+names[obj]+postfix) )
                    applyPostfix(process,"patDefaultSequence",postfix).remove(
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
                    ## check whether module is in sequence or not
                    result = [ m.label()[:-len(postfix)] for m in listModules( getattr(process,"patDefaultSequence"+postfix))]
                    result.extend([ m.label()[:-len(postfix)] for m in listSequences( getattr(process,"patDefaultSequence"+postfix))]  )
                    if applyPostfix(process,"patCandidateSummary",postfix) in result :
                        applyPostfix(process,"patCandidateSummary",postfix).candidates.remove(
                            cms.InputTag('pat'+names[obj]+postfix) )
                    if applyPostfix(process,"selectedPatCandidateSummary",postfix) in result :
                        applyPostfix(process,"selectedPatCandidateSummary",postfix).candidates.remove(
                            cms.InputTag('selectedPat'+names[obj]+postfix) )
                    if applyPostfix(process,"cleanPatCandidateSummary",postfix) in result :
                        applyPostfix(process,"cleanPatCandidateSummary",postfix).candidates.remove(
                            cms.InputTag('cleanPat'+names[obj]+postfix) )
        ## remove cleaning for the moment; in principle only the removed object
        ## could be taken out of the checkOverlaps PSet
        if len(outputModules) > 0:
            print "---------------------------------------------------------------------"
            print "INFO   : some objects have been removed from the sequence. Switching "
            print "         off PAT cross collection cleaning, as it might be of limited"
            print "         sense now. If you still want to keep object collection cross"
            print "         cleaning within PAT you need to run and configure it by hand"
            removeCleaning(process,outputModules=outputModules,postfix=postfix)

removeSpecificPATObjects=RemoveSpecificPATObjects()


class RemoveCleaning(ConfigToolBase):

    """ remove PAT cleaning from the default sequence:
    """
    _label='removeCleaning'
    _defaultParameters=dicttypes.SortedKeysDict()
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'outputModules',['out'], "names of all output modules specified to be adapted (default is ['out'])")
        self.addParameter(self._defaultParameters,'postfix',"", "postfix of default sequence")
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process,
                 outputInProcess = None,
                 postfix         = None,
                 outputModules   = None) :
        ## stop processing if 'outputInProcess' exists and show the new alternative
        if  not outputInProcess is None:
            deprecatedOptionOutputInProcess(self)
        if  outputModules is None:
            outputModules=self._defaultParameters['outputModules'].value
        if postfix  is None:
            postfix=self._defaultParameters['postfix'].value

        self.setParameter('outputModules',outputModules)
        self.setParameter('postfix',postfix)

        self.apply(process)

    def toolCode(self, process):
        outputModules=self._parameters['outputModules'].value
        postfix=self._parameters['postfix'].value

        ## adapt single object counters
        for m in listModules(applyPostfix(process,"countPatCandidates",postfix)):
            if hasattr(m, 'src'): m.src = m.src.value().replace('cleanPat','selectedPat')

        ## adapt lepton counter
        countLept = applyPostfix(process,"countPatLeptons",postfix)
        countLept.electronSource = countLept.electronSource.value().replace('cleanPat','selectedPat')
        countLept.muonSource = countLept.muonSource.value().replace('cleanPat','selectedPat')
        countLept.tauSource = countLept.tauSource.value().replace('cleanPat','selectedPat')
        for m in getattr(process, "cleanPatCandidates").moduleNames():
            getattr(process, "patDefaultSequence"+postfix).remove(
                applyPostfix(process,m,postfix)
                )
        if len(outputModules) > 0:
            print "------------------------------------------------------------"
            print "INFO   : cleaning has been removed. Switching output from"
            print "         clean PAT candidates to selected PAT candidates."
            ## add selected pat objects to the pat output
            from PhysicsTools.PatAlgos.patEventContent_cff import patEventContentNoCleaning
            for outMod in outputModules:
                if hasattr(process,outMod):
                    getattr(process,outMod).outputCommands = patEventContentNoCleaning
                else:
                    raise KeyError, "process has no OutModule named", outMod

removeCleaning=RemoveCleaning()


class AddCleaning(ConfigToolBase):

    """ Add PAT cleaning from the default sequence
    """
    _label='addCleaning'
    _defaultParameters=dicttypes.SortedKeysDict()
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'outputModules',['out'], "names of all output modules specified to be adapted (default is ['out'])")
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process,
                 outputInProcess = None,
                 outputModules   = None):
        ## stop processing if 'outputInProcess' exists and show the new alternative
        if  not outputInProcess is None:
            deprecatedOptionOutputInProcess(self)
        if  outputModules is None:
            outputModules=self._defaultParameters['outputModules'].value

        self.setParameter('outputModules',outputModules)
        self.apply(process)

    def toolCode(self, process):
        outputModules=self._parameters['outputModules'].value

        ## adapt single object counters
        process.patDefaultSequence.replace(process.countPatCandidates, process.cleanPatCandidates * process.countPatCandidates)
        for m in listModules(process.countPatCandidates):
            if hasattr(m, 'src'): m.src = m.src.value().replace('selectedPat','cleanPat')
        ## adapt lepton counter
        countLept = process.countPatLeptons
        countLept.electronSource = countLept.electronSource.value().replace('selectedPat','cleanPat')
        countLept.muonSource = countLept.muonSource.value().replace('selectedPat','cleanPat')
        countLept.tauSource = countLept.tauSource.value().replace('selectedPat','cleanPat')
        if len(outputModules) > 0:
            print "------------------------------------------------------------"
            print "INFO   : cleaning has been added. Switching output from  "
            print "         selected PAT candidates to clean PAT candidates."
            ## add clean layer1 objects to the pat output
            from PhysicsTools.PatAlgos.patEventContent_cff import patEventContent
            for outMod in outputModules:
                if hasattr(process,outMod):
                    getattr(process,outMod).outputCommands = patEventContent
                else:
                    raise KeyError, "process has no OutModule named", outMod

addCleaning=AddCleaning()

def deprecatedOptionOutputInProcess(obj):
    print "-------------------------------------------------------"
    print " INFO: the option 'outputInProcess' will be deprecated "
    print "       soon:", obj._label
    print "       please use the option 'outputModules' now and   "
    print "       specify the names of all needed OutModules in   "
    print "       there (default: ['out'])"
    print "-------------------------------------------------------"
    #raise KeyError, "Unsupported option 'outputInProcess' used in '"+obj._label+"'"
