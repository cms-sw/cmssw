from FWCore.GuiBrowsers.ConfigToolBase import *

from PhysicsTools.PatAlgos.tools.helpers import *

class RestrictInputToAOD(ConfigToolBase):

    """ Remove pat object production steps which rely on RECO event
    content
    """
    _label='RestrictInputToAOD'
    _defaultParameters={}
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'names',['All'], "list of collection names; supported are 'Photons', 'Electrons',, 'Muons', 'Taus', 'Jets', 'METs', 'All'", allowedValues=['Photons','Electrons', 'Muons', 'Taus', 'Jets', 'METs', 'All'])
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def dumpPython(self):
        dumpPythonImport = "\nfrom PhysicsTools.PatAlgos.tools.coreTools import *\n"
        dumpPython=''
        if self._comment!="":
            dumpPython = '#'+self._comment
        dumpPython = "\nrestrictInputToAOD(process, "
        dumpPython += str(self.getvalue('names'))+")"+'\n'
        return (dumpPythonImport,dumpPython) 

    def __call__(self,process,
                 names1     = None) :
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
    _label='RemoveMCMatching'
    _defaultParameters={}
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'names',['All'], "collection name; supported are 'Photons', 'Electrons','Muons', 'Taus', 'Jets', 'METs', 'All', 'PFAll', 'PFElectrons','PFTaus','PFMuons'", allowedValues=['Photons', 'Electrons','Muons', 'Taus', 'Jets', 'METs', 'All', 'PFAll', 'PFElectrons','PFTaus','PFMuons'])
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def dumpPython(self):
        dumpPythonImport = "\nfrom PhysicsTools.PatAlgos.tools.coreTools import *\n"
        dumpPython=''
        if self._comment!="":
            dumpPython = '#'+self._comment
        dumpPython = "\nremoveMCMatching(process, "
        dumpPython += str(self.getvalue('names'))+")"+'\n'
        return (dumpPythonImport,dumpPython) 

    def __call__(self,process,
                 names1     = None) :
        if  names is None:
            names=self._defaultParameters['names'].value
        self.setParameter('names',names)
        self.apply(process) 
        
    def toolCode(self, process):        
        names=self._parameters['names'].value

        print "************** MC dependence removal ************"
        for obj in range(len(names)):    
            if( names[obj] == 'Photons'   or names[obj] == 'All' ):
                print "removing MC dependencies for photons"
                _removeMCMatchingForPATObject(process, 'photonMatch', 'patPhotons') 
            if( names[obj] == 'Electrons' or names[obj] == 'All' ):
                print "removing MC dependencies for electrons"
                _removeMCMatchingForPATObject(process, 'electronMatch', 'patElectrons') 
            if( names[obj] == 'Muons'     or names[obj] == 'All' ):
                print "removing MC dependencies for muons"
                _removeMCMatchingForPATObject(process, 'muonMatch', 'patMuons') 
            if( names[obj] == 'Taus'      or names[obj] == 'All' ):
                print "removing MC dependencies for taus"
                _removeMCMatchingForPATObject(process, 'tauMatch', 'patTaus')
                ## remove mc extra modules for taus
                process.patDefaultSequence.remove(process.tauGenJets)
                process.patDefaultSequence.remove(process.tauGenJetMatch)
                ## remove mc extra configs for taus
                tauProducer = getattr(process, 'patTaus')
                tauProducer.addGenJetMatch      = False
                tauProducer.embedGenJetMatch    = False
                tauProducer.genJetMatch         = ''         
            if( names[obj] == 'Jets'      or names[obj] == 'All' ):
                print "removing MC dependencies for jets"
                ## remove mc extra modules for jets
                process.patDefaultSequence.remove(process.patJetPartonMatch)
                process.patDefaultSequence.remove(process.patJetGenJetMatch)
                process.patDefaultSequence.remove(process.patJetFlavourId)
                ## remove mc extra configs for jets
                jetProducer = getattr(process, jetCollectionString())
                jetProducer.addGenPartonMatch   = False
                jetProducer.embedGenPartonMatch = False
                jetProducer.genPartonMatch      = ''
                jetProducer.addGenJetMatch      = False
                jetProducer.genJetMatch         = ''
                jetProducer.getJetMCFlavour     = False
                jetProducer.JetPartonMapSource  = ''       
            if( names[obj] == 'METs'      or names[obj] == 'All' ):
                ## remove mc extra configs for jets
                metProducer = getattr(process, 'patMETs')        
                metProducer.addGenMET           = False
                metProducer.genMETSource        = ''       
            if( names[obj] == 'PFElectrons' or names[obj] == 'PFAll' ):
                print "now removing MC dependencies for PF electrons"
                _removeMCMatchingForPATObject(process, 'electronMatch', 'patElectrons') 
            if( names[obj] == 'PFMuons'     or names[obj] == 'PFAll' ):
                print "now removing MC dependencies for PF muons"
                _removeMCMatchingForPATObject(process, 'muonMatch', 'patMuons') 
            if( names[obj] == 'PFTaus'      or names[obj] == 'PFAll' ):
                print "now removing MC dependencies for PF taus"
                _removeMCMatchingForPATObject(process, 'tauMatch', 'patTaus')
                process.patDefaultSequence.remove(process.tauGenJetMatch)
                process.patDefaultSequence.remove(process.tauGenJets)
                ## remove mc extra configs for taus
                tauProducer = getattr(process, 'patTaus')
                tauProducer.addGenJetMatch      = False
                tauProducer.embedGenJetMatch    = False
                tauProducer.genJetMatch         = ''         
            if( names[obj] == 'PFJets'      or names[obj] == 'PFAll' ):
                print "now removing MC dependencies for PF jets"
                ## remove mc extra modules for jets
                process.patDefaultSequence.remove(process.patJetPartons)
                process.patDefaultSequence.remove(process.pfPatJetPartonMatch)
                process.patDefaultSequence.remove(process.pfPatJetGenJetMatch)
                process.patDefaultSequence.remove(process.pfPatJetPartonAssociation)
                process.patDefaultSequence.remove(process.pfPatJetFlavourAssociation)     
                ## remove mc extra configs for jets
                jetProducer = getattr(process, 'pfPatJets')
                jetProducer.addGenPartonMatch   = False
                jetProducer.embedGenPartonMatch = False
                jetProducer.genPartonMatch      = ''
                jetProducer.addGenJetMatch      = False
                jetProducer.genJetMatch         = ''
                jetProducer.getJetMCFlavour     = False
                jetProducer.JetPartonMapSource  = ''       
            if( names[obj] == 'PFMETs'      or names[obj] == 'PFAll' ):
                print "now removing MC dependencies for PF MET"
                ## remove mc extra configs for jets
                metProducer = getattr(process, 'pfMET')        
                metProducer.addGenMET           = cms.bool(False)
                metProducer.genMETSource        = cms.InputTag('')
            
removeMCMatching=RemoveMCMatching()

def _removeMCMatchingForPATObject(process, matcherName, producerName):
    ## remove mcMatcher from the default sequence
    objectMatcher = getattr(process, matcherName)
    if (producerName=='pfPatMuons'or producerName=='pfPatTaus'):
        process.PFPATafterPAT.remove(objectMatcher)
    if (producerName=='patMuons'or producerName=='patTaus'or
        producerName=='patPhotons' or producerName=='patElectrons'):
        process.patDefaultSequence.remove(objectMatcher)
    ## straighten photonProducer
    objectProducer = getattr(process, producerName)
    objectProducer.addGenMatch      = False
    objectProducer.embedGenMatch    = False
    objectProducer.genParticleMatch = ''
    
    
class removeAllPATObjectsBut(ConfigToolBase):

    """ Remove all PAT objects from the default sequence but a specific
    one
    """
    _label='removeAllPATObjectsBut'
    _defaultParameters={}
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'names',self._defaultValue, "list of collection names; supported are 'Photons', 'Electrons', 'Muons', 'Taus', 'Jets', 'METs'", Type=list, allowedValues=['Photons', 'Electrons', 'Muons', 'Taus', 'Jets', 'METs'])
        self.addParameter(self._defaultParameters,'outputInProcess',True, "indicate whether there is an output module specified for the process (default is True)  ")
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def dumpPython(self):
        dumpPythonImport = "\nfrom PhysicsTools.PatAlgos.tools.coreTools import *\n"
        dumpPython=''
        if self._comment!="":
            dumpPython = '#'+self._comment
        dumpPython = "\nremoveAllPATObjectsBut(process, "
        dumpPython += str(self.getvalue('names'))+", "
        dumpPython += str(self.getvalue('outputInProcess'))+")"+'\n'
        return (dumpPythonImport,dumpPython) 

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
       
removeAllPATObjectsBut=removeAllPATObjectsBut()


class RemoveSpecificPATObjects(ConfigToolBase):

    """ Remove a specific PAT object from the default sequence:
    """
    _label='RemoveSpecificPATObjects'
    _defaultParameters={}
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'names',self._defaultValue, "list of collection names; supported are 'Photons', 'Electrons', 'Muons', 'Taus', 'Jets', 'METs'", Type=list, allowedValues=['Photons', 'Electrons', 'Muons', 'Taus', 'Jets', 'METs'])
        self.addParameter(self._defaultParameters,'outputInProcess',True,"indicate whether there is an output module specified for the process (default is True)" )
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def dumpPython(self):
        dumpPythonImport = "\nfrom PhysicsTools.PatAlgos.tools.coreTools import *\n"
        dumpPython=''
        if self._comment!="":
            dumpPython = '#'+self._comment
        dumpPython = "\nremoveSpecificPATObjects(process, "
        dumpPython += str(self.getvalue('names'))+", "
        dumpPython += str(self.getvalue('outputInProcess'))+")"+'\n'
        return (dumpPythonImport,dumpPython) 

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
        ## remove pre object production steps from the default sequence

        for obj in range(len(names)):
            if( names[obj] == 'Photons' ):
                process.patDefaultSequence.remove(getattr(process, 'patPhotonIsolation'))
                process.patDefaultSequence.remove(getattr(process, 'photonMatch'))
            if( names[obj] == 'Electrons' ):
                process.patDefaultSequence.remove(getattr(process, 'patElectronId'))
                process.patDefaultSequence.remove(getattr(process, 'patElectronIsolation'))
                process.patDefaultSequence.remove(getattr(process, 'electronMatch'))
            if( names[obj] == 'Muons' ):
                process.patDefaultSequence.remove(getattr(process, 'muonMatch'))
            if( names[obj] == 'Taus' ):
                process.patDefaultSequence.remove(getattr(process, 'patPFCandidateIsoDepositSelection'))
                process.patDefaultSequence.remove(getattr(process, 'patPFTauIsolation'))
                process.patDefaultSequence.remove(getattr(process, 'tauMatch'))
                process.patDefaultSequence.remove(getattr(process, 'tauGenJets'))
                process.patDefaultSequence.remove(getattr(process, 'tauGenJetMatch'))
            if( names[obj] == 'Jets' ):
                print process.patDefaultSequence
                process.patDefaultSequence.remove(getattr(process, 'patJetCharge'))
                process.patDefaultSequence.remove(getattr(process, 'patJetCorrections'))
                process.patDefaultSequence.remove(getattr(process, 'patJetPartonMatch'))
                process.patDefaultSequence.remove(getattr(process, 'patJetGenJetMatch'))
                process.patDefaultSequence.remove(getattr(process, 'patJetFlavourId'))
            if( names[obj] == 'METs' ):
                process.patDefaultSequence.remove(getattr(process, 'patMETCorrections'))
        
            ## remove object production steps from the default sequence    
            if( names[obj] == 'METs' ):
                process.patCandidates.remove( getattr(process, 'pat'+names[obj]) )
            else:
                if( names[obj] == 'Jets' ):
                    process.patCandidates.remove( getattr(process, jetCollectionString()) )
                    process.selectedPatCandidates.remove( getattr(process, jetCollectionString('selected')) )
                    process.countPatCandidates.remove( getattr(process, jetCollectionString('count')) )
                else:
                    process.patCandidates.remove( getattr(process, 'pat'+names[obj]) )
                    process.selectedPatCandidates.remove( getattr(process, 'selectedPat'+names[obj]) )
                    process.countPatCandidates.remove( getattr(process, 'countPat'+names[obj]) )
            ## in the case of leptons, the lepton counter must be modified as well
            if( names[obj] == 'Electrons' ):
                print 'removed from lepton counter: electrons'
                process.countPatLeptons.countElectrons = False
            elif( names[obj] == 'Muons' ):
                print 'removed from lepton counter: muons'
                process.countPatLeptons.countMuons = False
            elif( names[obj] == 'Taus' ):
                print 'removed from lepton counter: taus'
                process.countPatLeptons.countTaus = False
            ## remove from summary
            if( names[obj] == 'METs' ):
                process.patCandidateSummary.candidates.remove( cms.InputTag('pat'+names[obj]) )
            else:
                if( names[obj] == 'Jets' ):
                    process.patCandidateSummary.candidates.remove( cms.InputTag(jetCollectionString()) )
                    process.selectedPatCandidateSummary.candidates.remove( cms.InputTag(jetCollectionString('selected')) )
                    process.cleanPatCandidateSummary.candidates.remove( cms.InputTag(jetCollectionString('clean')) )
                else:
                    process.patCandidateSummary.candidates.remove( cms.InputTag('pat'+names[obj]) )
                    process.selectedPatCandidateSummary.candidates.remove( cms.InputTag('selectedPat'+names[obj]) )
                    process.cleanPatCandidateSummary.candidates.remove( cms.InputTag('cleanPat'+names[obj]) )
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

    """ Remove PAT cleaning from the default sequence:
    """
    _label='RemoveCleaning'
    _defaultParameters={}
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'outputInProcess',True,"indicate whether there is an output module specified for the process (default is True)" )
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def dumpPython(self):
        dumpPythonImport = "\nfrom PhysicsTools.PatAlgos.tools.coreTools import *\n"
        dumpPython=''
        if self._comment!="":
            dumpPython = '#'+self._comment
        dumpPython = "\nremoveCleaning(process, "
        dumpPython += str(self.getvalue('outputInProcess'))+")"+'\n'
        return (dumpPythonImport,dumpPython) 

    def __call__(self,process,
                 outputInProcess = None) :
        if  outputInProcess is None:
            outputInProcess=self._defaultParameters['outputInProcess'].value
        self.setParameter('outputInProcess',outputInProcess)
        self.apply(process) 
        
    def toolCode(self, process):        
        outputInProcess=self._parameters['outputInProcess'].value

        ## adapt single object counters
        for m in listModules(process.countPatCandidates):
            if hasattr(m, 'src'): m.src = m.src.value().replace('cleanPat','selectedPat')
        ## adapt lepton counter
        countLept = process.countPatLeptons
        countLept.electronSource = countLept.electronSource.value().replace('cleanPat','selectedPat')
        countLept.muonSource = countLept.muonSource.value().replace('cleanPat','selectedPat')
        countLept.tauSource = countLept.tauSource.value().replace('cleanPat','selectedPat')
        process.patDefaultSequence.remove(process.cleanPatCandidates)
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
    _label='AddCleaning'
    _defaultParameters={}
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'outputInProcess',True, "")
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def dumpPython(self):
        dumpPythonImport = "\nfrom PhysicsTools.PatAlgos.tools.coreTools import *\n"
        dumpPython=''
        if self._comment!="":
            dumpPython = '#'+self._comment
        dumpPython = "\naddCleaning(process, "
        dumpPython += '"'+str(self.getvalue('outputInProcess'))+")"+'\n'
        return (dumpPythonImport,dumpPython) 

    def __call__(self,process,
                 outputInProcess     = None):
        if  outputInProcess is None:
            outputInProcess=self._defaultParameters['outputInProcess'].value
        
        self.setParameter('outputInProcess',outputInProcess)
        self.apply(process) 
        
    def toolCode(self, process):        
        outputInProcess=self._parameters['outputInProcess'].value

        ## adapt single object counters
        process.patDefaultSequence.replace(process.countPatCandidates, process.cleanPatCandidates * process.countPatCandidates)
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
