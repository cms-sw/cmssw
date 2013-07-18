from FWCore.GuiBrowsers.ConfigToolBase import *
from PhysicsTools.PatAlgos.tools.helpers import *


class ConfigureHeavyIons(ConfigToolBase):

    """ Configure all defaults for heavy ions
    """
    _label='configureHeavyIons'
    _defaultParameters=dicttypes.SortedKeysDict()
    
    def __init__(self):
        ConfigToolBase.__init__(self)        
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process):
                
        self.apply(process) 
        
    def toolCode(self, process):        
        productionDefaults(process)
        selectionDefaults(process)
           
configureHeavyIons=ConfigureHeavyIons()


class ProductionDefaults(ConfigToolBase):

    """ Configure all relevant layer1 candidates for heavy ions
    """
    _label='productionDefaults'
    _defaultParameters=dicttypes.SortedKeysDict()
    
    def __init__(self):
        ConfigToolBase.__init__(self)        
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters
    
    def __call__(self,process):
                
        self.apply(process) 
        
    def toolCode(self, process):        
        ## adapt jet defaults
        patJets = getattr(process, jetCollectionString())
        patJets.jetSource  = cms.InputTag("iterativeConePu5CaloJets")

        jetCors  = getattr(process, 'patJetCorrFactors')
        jetCors.jetSource = cms.InputTag("iterativeConePu5CaloJets")
        jetCors.corrLevels = cms.PSet(L2Relative = cms.string("L2Relative_IC5Calo"),
                                      L3Absolute = cms.string("L3Absolute_IC5Calo"),
                                      L1Offset   = cms.string('none'),
                                      L4EMF      = cms.string('none'),
                                      L5Flavor   = cms.string('none'),
                                      L6UE       = cms.string('none'),
                                      L7Parton   = cms.string('none') 
                                      )

        partonMatch = getattr(process, 'patJetPartonMatch')
        partonMatch.src = cms.InputTag("iterativeConePu5CaloJets")
        partonMatch.matched = cms.InputTag("hiPartons")

        jetMatch = getattr(process, 'patJetGenJetMatch')
        jetMatch.src     = cms.InputTag("iterativeConePu5CaloJets")
        jetMatch.matched = cms.InputTag("heavyIonCleanedGenJets")
        
        patJets.addBTagInfo         = False
        patJets.addTagInfos         = False
        patJets.addDiscriminators   = False
        patJets.addAssociatedTracks = False
        patJets.addJetCharge        = False
        patJets.addJetID            = False
        patJets.getJetMCFlavour     = False
        patJets.addGenPartonMatch   = True
        patJets.addGenJetMatch      = True
        patJets.embedGenJetMatch    = True
        patJets.embedGenPartonMatch   = True

        ## adapt muon defaults
        muonMatch = getattr(process, 'muonMatch')
        muonMatch.matched = cms.InputTag("hiGenParticles")
        patMuons  = getattr(process, 'patMuons')
        patMuons.embedGenMatch = cms.bool(True)
        process.patMuons.embedCaloMETMuonCorrs = cms.bool(False)
        process.patMuons.embedTcMETMuonCorrs   = cms.bool(False)
        process.patMuons.embedPFCandidate   = cms.bool(False)
        process.patMuons.useParticleFlow    = cms.bool(False)
        process.patMuons.addEfficiencies    = cms.bool(False)
        process.patMuons.addResolutions     = cms.bool(False)
        process.patMuons.pvSrc = cms.InputTag("hiSelectedVertex")
        
        ## adapt photon defaults
        photonMatch = getattr(process, 'photonMatch')
        photonMatch.matched = cms.InputTag("hiGenParticles")
        patPhotons  = getattr(process, 'patPhotons')
        patPhotons.addPhotonID   = cms.bool(True)
        patPhotons.addGenMatch   = cms.bool(True)
        patPhotons.embedGenMatch = cms.bool(True)
        patPhotons.userData.userFloats.src  = cms.VInputTag(
            cms.InputTag( "isoCC1"),cms.InputTag( "isoCC2"),cms.InputTag( "isoCC3"),cms.InputTag( "isoCC4"),cms.InputTag("isoCC5"),
            cms.InputTag( "isoCR1"),cms.InputTag( "isoCR2"),cms.InputTag( "isoCR3"),cms.InputTag( "isoCR4"),cms.InputTag("isoCR5"),
            cms.InputTag( "isoT11"),cms.InputTag( "isoT12"),cms.InputTag( "isoT13"),cms.InputTag( "isoT14"),  
            cms.InputTag( "isoT21"),cms.InputTag( "isoT22"),cms.InputTag( "isoT23"),cms.InputTag( "isoT24"),  
            cms.InputTag( "isoT31"),cms.InputTag( "isoT32"),cms.InputTag( "isoT33"),cms.InputTag( "isoT34"),  
            cms.InputTag( "isoT41"),cms.InputTag( "isoT42"),cms.InputTag( "isoT43"),cms.InputTag( "isoT44"),  
            cms.InputTag("isoDR11"),cms.InputTag("isoDR12"),cms.InputTag("isoDR13"),cms.InputTag("isoDR14"),  
            cms.InputTag("isoDR21"),cms.InputTag("isoDR22"),cms.InputTag("isoDR23"),cms.InputTag("isoDR24"),  
            cms.InputTag("isoDR31"),cms.InputTag("isoDR32"),cms.InputTag("isoDR33"),cms.InputTag("isoDR34"),  
            cms.InputTag("isoDR41"),cms.InputTag("isoDR42"),cms.InputTag("isoDR43"),cms.InputTag("isoDR44")
            )
        patPhotons.photonIDSource = cms.InputTag("PhotonIDProd","PhotonCutBasedIDLoose")
        del patPhotons.photonIDSources
        
productionDefaults=ProductionDefaults()


class SelectionDefaults(ConfigToolBase):

    """ Configure all relevant selected layer1 candidates for heavy ions
    """
    _label='selectionDefaults'
    _defaultParameters=dicttypes.SortedKeysDict()
    
    def __init__(self):
        ConfigToolBase.__init__(self)        
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process):
                
        self.apply(process) 
        
    def toolCode(self, process):        
        selectedJets = getattr(process, jetCollectionString('selected'))
        selectedJets.cut = cms.string('pt > 20.')
        selectedMuons = getattr(process, 'selectedPatMuons')
        selectedMuons.cut = cms.string('pt > 0. & abs(eta) < 12.')
        selectedPhotons = getattr(process, 'selectedPatPhotons')
        selectedPhotons.cut = cms.string('pt > 0. & abs(eta) < 12.')
        
selectionDefaults=SelectionDefaults()


class DisbaleMonteCarloDeps(ConfigToolBase):

    """ Cut off all MC dependencies
    """
    _label='disableMonteCarloDeps'
    _defaultParameters=dicttypes.SortedKeysDict()
    
    def __init__(self):
        ConfigToolBase.__init__(self)        
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self,process):
                
        self.apply(process) 
        
    def toolCode(self, process):
        ## switch MC to false in heavyIon Producer
        process.heavyIon.doMC = False
        
        ## remove MC matching from heavyIonJets
        process.makeHeavyIonJets.remove(process.genPartons)
        process.makeHeavyIonJets.remove(process.heavyIonCleanedGenJets)
        process.makeHeavyIonJets.remove(process.hiPartons)
        process.makeHeavyIonJets.remove(process.patJetGenJetMatch)
        process.makeHeavyIonJets.remove(process.patJetPartonMatch)
        
        process.patJets.addGenPartonMatch   = False
        process.patJets.embedGenPartonMatch = False
        process.patJets.genPartonMatch      = ''
        process.patJets.addGenJetMatch      = False
        process.patJets.genJetMatch	      = ''
        process.patJets.getJetMCFlavour     = False
        process.patJets.JetPartonMapSource  = ''

        ## remove MC matching from heavyIonMuons        
        process.makeHeavyIonMuons.remove(process.muonMatch)
        
        process.patMuons.addGenMatch        = False
        process.patMuons.embedGenMatch      = False
        
disableMonteCarloDeps=DisbaleMonteCarloDeps()       
