from __future__ import print_function
from builtins import range
from PhysicsTools.PatAlgos.tools.ConfigToolBase import *

from PhysicsTools.PatAlgos.tools.helpers import *

class RunOnData(ConfigToolBase):

    """ Remove monte carlo matching from a given collection or all PAT
    candidate collections and adapt the JEC's:
    """
    _label='runOnData'
    _defaultParameters=dicttypes.SortedKeysDict()
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'names',['All'], "collection name; supported are 'Photons', 'Electrons', 'LowPtElectrons', 'Muons', 'Taus', 'TausBoosted', 'Jets', 'METs', 'All', 'PFAll', 'PFElectrons','PFTaus','PFMuons'", allowedValues=['Photons', 'Electrons', 'LowPtElectrons', 'Muons', 'Taus', 'TausBoosted', 'Jets', 'METs', 'All', 'PFAll', 'PFElectrons','PFTaus','PFMuons'])
        self.addParameter(self._defaultParameters,'postfix',"", "postfix of default sequence")
        self.addParameter(self._defaultParameters,'outputModules',['out'], "names of all output modules specified to be adapted (default is ['out'])")
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process,
                 names           = None,
                 postfix         = None,
                 outputModules   = None) :
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

        print('******************* RunOnData *******************')
        removeMCMatching(process, names=names, postfix=postfix, outputModules=outputModules)
        for mod in process.producerNames().split():
            if mod.startswith('patJetCorrFactors'):
                prefix = getattr(process, mod).payload.pythonValue().replace("'","")
                if 'L3Absolute' in getattr(process,mod).levels:
                    if not 'L2L3Residual' in getattr(process,mod).levels:
                        getattr(process,mod).levels.insert(getattr(process,mod).levels.index('L3Absolute')+1, 'L2L3Residual')
                        print('adding L2L3Residual JEC for:', getattr(process,mod).label_())
                if hasattr(process, prefix+'CombinedCorrector'+postfix):
                    if prefix+'L3Absolute' in getattr(process,prefix+'CombinedCorrector'+postfix).correctors:
                        if not prefix+'L2L3Residual' in getattr(process,prefix+'CombinedCorrector'+postfix).correctors:
                            idx = getattr(process,prefix+'CombinedCorrector'+postfix).correctors.index(prefix+'L3Absolute')+1
                            getattr(process,prefix+'CombinedCorrector'+postfix).correctors.insert(idx, prefix+'L2L3Residual')
                            print('adding L2L3Residual for TypeI MET correction:', getattr(process,prefix+'CombinedCorrector'+postfix).label_())

runOnData=RunOnData()


class RemoveMCMatching(ConfigToolBase):

    """ Remove monte carlo matching from a given collection or all PAT
    candidate collections:
    """
    _label='removeMCMatching'
    _defaultParameters=dicttypes.SortedKeysDict()
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters,'names',['All'], 
                          "collection name; supported are 'Photons', 'OOTPhotons', 'Electrons', 'LowPtElectrons', 'Muons', 'Taus', 'TausBoosted', 'Jets', 'METs', 'All', 'PFAll', 'PFElectrons','PFTaus','PFMuons'", 
                          allowedValues=['Photons', 'OOTPhotons' ,'Electrons', 'LowPtElectrons', 'Muons', 'Taus', 'TausBoosted', 'Jets', 'METs', 'All', 'PFAll', 'PFElectrons','PFTaus','PFMuons'])
        self.addParameter(self._defaultParameters,'postfix',"", "postfix of default sequence")
        self.addParameter(self._defaultParameters,'outputModules',['out'], "names of all output modules specified to be adapted (default is ['out'])")
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process,
                 names           = None,
                 postfix         = None,
                 outputModules   = None) :
        if  names is None:
            names=self._defaultParameters['names'].value
        if postfix  is None:
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

        print("************** MC dependence removal ************")
        attrsToDelete = []
        for obj in range(len(names)):
            if( names[obj] == 'Photons'   or names[obj] == 'All' ):
                print("removing MC dependencies for photons")
                _removeMCMatchingForPATObject(process, 'photonMatch', 'patPhotons', postfix)
            if( names[obj] == 'OOTPhotons'   or names[obj] == 'All' ):
                print("removing MC dependencies for out of time photons")
                _removeMCMatchingForPATObject(process, 'ootPhotonMatch', 'patOOTPhotons', postfix)
            if( names[obj] == 'Electrons' or names[obj] == 'All' ):
                print("removing MC dependencies for electrons")
                _removeMCMatchingForPATObject(process, 'electronMatch', 'patElectrons', postfix)
            if( names[obj] == 'LowPtElectrons' or names[obj] == 'All' ):
                print("removing MC dependencies for LowPtElectrons")
                _removeMCMatchingForPATObject(process, 'lowPtElectronMatch', 'patLowPtElectrons', postfix)
            if( names[obj] == 'Muons'     or names[obj] == 'All' ):
                print("removing MC dependencies for muons")
                _removeMCMatchingForPATObject(process, 'muonMatch', 'patMuons', postfix)
            if( names[obj] == 'Taus'      or names[obj] == 'All' ):
                print("removing MC dependencies for taus")
                _removeMCMatchingForPATObject(process, 'tauMatch', 'patTaus', postfix)
                ## remove mc extra configs for taus
                tauProducer = getattr(process,'patTaus'+postfix)
                tauProducer.addGenJetMatch   = False
                tauProducer.embedGenJetMatch = False
                attrsToDelete += [tauProducer.genJetMatch.getModuleLabel()]
                tauProducer.genJetMatch      = ''
                attrsToDelete += ['tauGenJets'+postfix]
                attrsToDelete += ['tauGenJetsSelectorAllHadrons'+postfix]
            #Boosted Taus
            if( names[obj] == 'TausBoosted'      or names[obj] == 'All' ):
                print("removing MC dependencies for taus boosted %s" %postfix)
                if hasattr(process, 'tauMatchBoosted'+postfix) and hasattr(process, 'patTausBoosted'+postfix) :
                    _removeMCMatchingForPATObject(process, 'tauMatchBoosted', 'patTausBoosted', postfix)
                    ## remove mc extra configs for taus
                    tauProducer = getattr(process,'patTausBoosted'+postfix)
                    tauProducer.addGenJetMatch   = False
                    tauProducer.embedGenJetMatch = False
                    attrsToDelete += [tauProducer.genJetMatch.getModuleLabel()]
                    tauProducer.genJetMatch      = ''
                    attrsToDelete += ['tauGenJetsBoosted'+postfix]
                    attrsToDelete += ['tauGenJetsSelectorAllHadronsBoosted'+postfix]
                else :
                    print("...skipped since taus boosted %s" %postfix, "are not part of process.")  
            if( names[obj] == 'Jets'      or names[obj] == 'All' ):
                print("removing MC dependencies for jets")
                jetPostfixes = []
                for mod in process.producerNames().split():
                    if mod.startswith('patJets') and getattr(process,mod).type_() == "PATJetProducer":
                        jetPostfixes.append(getattr(process, mod).label_().replace("patJets",""))
                for pfix in jetPostfixes:
                    ## remove mc extra configs for jets
                    jetProducer = getattr(process, jetCollectionString()+pfix)
                    jetProducer.addGenPartonMatch   = False
                    jetProducer.embedGenPartonMatch = False
                    #attrsToDelete += [jetProducer.genPartonMatch.getModuleLabel()] #MM needed for potential jet backuping
                    jetProducer.genPartonMatch      = ''
                    jetProducer.addGenJetMatch      = False
                    #attrsToDelete += [jetProducer.genJetMatch.getModuleLabel()]  #MM needed for potential jet backuping
                    jetProducer.genJetMatch         = ''
                    jetProducer.getJetMCFlavour     = False
                    jetProducer.useLegacyJetMCFlavour = False
                    jetProducer.addJetFlavourInfo   = False
                    #attrsToDelete += [jetProducer.JetPartonMapSource.getModuleLabel()]  #MM needed for potential jet backuping
                    jetProducer.JetPartonMapSource  = ''
                    #attrsToDelete += [jetProducer.JetFlavourInfoSource.getModuleLabel()]  #MM needed for potential jet backuping
                    jetProducer.JetFlavourInfoSource = ''
                    attrsToDelete += ['slimmedGenJets'+pfix]
                ## adjust output
                for outMod in outputModules:
                    if hasattr(process,outMod):
                        getattr(process,outMod).outputCommands.append("drop *_selectedPatJets*_genJets_*")
                        getattr(process,outMod).outputCommands.append("drop recoGenJets_*_*_*")
                    else:
                        raise KeyError("process has no OutModule named " + outMod)

            if( names[obj] == 'METs'      or names[obj] == 'All' ):
                for mod in process.producerNames().split():
                    if mod.startswith('pat') and getattr(process,mod).type_() == "PATMETProducer":
                        ## remove mc extra configs for MET
                        metProducer = getattr(process, mod)
                        metProducer.addGenMET           = False
                        attrsToDelete += [metProducer.genMETSource.getModuleLabel()]
                        metProducer.genMETSource        = ''
        attrsToDelete += [
            'prunedGenParticles',
            'prunedGenParticlesWithStatusOne',
            'packedGenParticles',
        ]
        for attr in attrsToDelete:
            if hasattr(process,attr): delattr(process,attr)

removeMCMatching=RemoveMCMatching()

def _removeMCMatchingForPATObject(process, matcherName, producerName, postfix=""):
    if hasattr(process, matcherName+postfix):
        objectMatcher = getattr(process, matcherName+postfix)
    if hasattr(process, producerName+postfix):
        objectProducer = getattr(process, producerName+postfix)
        objectProducer.addGenMatch      = False
        objectProducer.embedGenMatch    = False
        attr = objectProducer.genParticleMatch.getModuleLabel()
        objectProducer.genParticleMatch = ''
    if hasattr(process,attr): delattr(process,attr)

