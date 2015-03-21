import FWCore.ParameterSet.Config as cms

from FWCore.GuiBrowsers.ConfigToolBase import *

import PhysicsTools.PatAlgos.tools.helpers as configtools
from PhysicsTools.PatAlgos.tools.trigTools import _addEventContent

#from PhysicsTools.PatUtils.patPFMETCorrections_cff import *
import RecoMET.METProducers.METSigParams_cfi as jetResolutions
from PhysicsTools.PatAlgos.producersLayer1.metProducer_cfi import patMETs

from PhysicsTools.PatUtils.tools.objectsUncertaintyTools import *

##MM
#from PhysicsTools.PatAlgos.patSequences_cff import *

class JetMEtUncertaintyTools(ConfigToolBase):

    """ Base class for estimating systematic uncertainties on MET """
    _label='JetMEtUncertaintyTools'
    _defaultParameters=dicttypes.SortedKeysDict()
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters, 'electronCollection', cms.InputTag('cleanPatElectrons'),
	                  "Input electron collection", Type=cms.InputTag, acceptNoneValue=True)
	self.addParameter(self._defaultParameters, 'photonCollection', None, # CV: set to empty InputTag to avoid double-counting wrt. cleanPatElectrons collection
	                  "Input photon collection", Type=cms.InputTag, acceptNoneValue=True)
	self.addParameter(self._defaultParameters, 'muonCollection', cms.InputTag('cleanPatMuons'),
                          "Input muon collection", Type=cms.InputTag, acceptNoneValue=True)
	self.addParameter(self._defaultParameters, 'tauCollection', cms.InputTag('cleanPatTaus'),
                          "Input tau collection", Type=cms.InputTag, acceptNoneValue=True)
	self.addParameter(self._defaultParameters, 'jetCollection', cms.InputTag('cleanPatJets'),
                          "Input jet collection", Type=cms.InputTag)
        self.addParameter(self._defaultParameters, 'jetCorrLabel', cms.InputTag("L3Absolute"),
                          "NOTE: use 'L3Absolute' for MC/'L2L3Residual' for Data", Type=cms.InputTag)
	self.addParameter(self._defaultParameters, 'doSmearJets', True,
                          "Flag to enable/disable jet smearing to better match MC to Data", Type=bool)
	self.addParameter(self._defaultParameters, 'jetSmearFileName', 'PhysicsTools/PatUtils/data/pfJetResolutionMCtoDataCorrLUT.root',
                          "Name of ROOT file containing histogram with jet smearing factors", Type=str)
        self.addParameter(self._defaultParameters, 'jetSmearHistogram', 'pfJetResolutionMCtoDataCorrLUT',
                          "Name of histogram with jet smearing factors", Type=str)
	self.addParameter(self._defaultParameters, 'jetCorrPayloadName', 'AK4PF',
                          "Use AK4PF for PFJets, AK4Calo for CaloJets", Type=str)
        self.addParameter(self._defaultParameters, 'jetCorrLabelUpToL3', cms.InputTag('ak4PFL1FastL2L3Corrector'),
                          "Use ak4PFL1FastL2L3Corrector (ak4PFchsL1FastL2L3Corrector) for PFJets with (without) charged hadron subtraction, ak4CaloL1FastL2L3Corrector for CaloJets", Type=cms.InputTag)
        self.addParameter(self._defaultParameters, 'jetCorrLabelUpToL3Res', cms.InputTag('ak4PFL1FastL2L3ResidualCorrector'),
                          "Use ak4PFL1FastL2L3ResidualCorrector (ak4PFchsL1FastL2L3ResiduaCorrectorl) for PFJets with (without) charged hadron subtraction, ak4CaloL1FastL2L3ResidualCorrector for CaloJets", Type=cms.InputTag)
        self.addParameter(self._defaultParameters, 'jecUncertaintyFile', "PhysicsTools/PatUtils/data/Summer13_V1_DATA_UncertaintySources_AK5PF.txt",
                          "Name of file containing jet energy uncertainty parameters", Type=str)
        self.addParameter(self._defaultParameters, 'jecUncertaintyTag', 'SubTotalMC',
                          "Name of tag for Data/MC jet energy uncertainties", Type=str)
	self.addParameter(self._defaultParameters, 'varyByNsigmas', 1.0,
                          "Number of standard deviations by which energies are varied", Type=float)
        self.addParameter(self._defaultParameters, 'addToPatDefaultSequence', True,
                          "Flag to enable/disable that metUncertaintySequence is inserted into patDefaultSequence", Type=bool)
        self.addParameter(self._defaultParameters, 'outputModule', 'out',
                          "Module label of PoolOutputModule (empty label indicates no PoolOutputModule is to be configured)", Type=str)
        self.addParameter(self._defaultParameters, 'postfix', '',
                          "Technical parameter to identify the resulting sequence and its modules (allows multiple calls in a job)", Type=str)
        self._parameters = copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def getDefaultParameters(self):
        return self._defaultParameters


    def _initializeInputTag(self, input, default):
        retVal = None
        if input is None:
            retVal = self._defaultParameters[default].value
        elif type(input) == str:
            retVal = cms.InputTag(input)
        else:
            retVal = input
        return retVal


    def _addCleanedJets(self, process, jetCollection,
                   electronCollection, photonCollection,
                   muonCollection, tauCollection,
                   uncertaintySequence, postfix = ""):

        # produce collection of jets not overlapping with reconstructed
        # electrons/photons, muons and tau-jet candidates
        jetsNotOverlappingWithLeptons = cms.EDProducer("PATJetCleaner",
            src = jetCollection,
            preselection = cms.string(''),
            checkOverlaps = cms.PSet(),
            finalCut = cms.string('')
        )
        numOverlapCollections = 0
        radius = 0.5
        if "ak4" in jetCollection.moduleLabel.lower(): radius=0.4
        for collection in [
            [ 'electrons', electronCollection ],
            [ 'photons',   photonCollection   ],
            [ 'muons',     muonCollection     ],
            [ 'taus',      tauCollection      ] ]:
            if isValidInputTag(collection[1]):
                setattr(jetsNotOverlappingWithLeptons.checkOverlaps, collection[0], cms.PSet(
                    src                 = collection[1],
                    algorithm           = cms.string("byDeltaR"),
                    preselection        = cms.string(""),
                    deltaR              = cms.double(radius),
                    checkRecoComponents = cms.bool(False),
                    pairCut             = cms.string(""),
                    requireNoOverlaps   = cms.bool(True),
                ))
                numOverlapCollections = numOverlapCollections + 1
        lastJetCollection = jetCollection.value()
        if numOverlapCollections >= 1:
            lastJetCollection = \
                addModuleToSequence(process, jetsNotOverlappingWithLeptons,
                                        [ jetCollection.value(), "NotOverlappingWithLeptonsForJetMEtUncertainty" ],
                                        uncertaintySequence, postfix)
        cleanedJetCollection = lastJetCollection

        return ( lastJetCollection, cleanedJetCollection )


    def _addShiftedParticleCollections(self, process,
                                       electronCollection = None,
                                       photonCollection = None,
                                       muonCollection = None,
                                       tauCollection = None,
                                       jetCollection = None, cleanedJetCollection = None,
                                       lastJetCollection = None, addShiftedResJetCollections = False,
                                       jetCorrLabelUpToL3 = None, jetCorrLabelUpToL3Res = None,
                                       jecUncertaintyFile = None, jecUncertaintyTag = None,
                                       jetSmearFileName=None, jetSmearHistogram=None,
                                       varyByNsigmas = None,
                                       postfix = ""):

        shiftedParticleSequence = cms.Sequence()
        shiftedParticleCollections = {}
        collectionsToKeep = []

        # standard jet collections
        shiftedParticleCollections[ 'cleanedJetCollection' ] = cleanedJetCollection

        #--------------------------------------------------------------------------------------------
        # store collection of jets shifted up/down in energy resolution
        #--------------------------------------------------------------------------------------------
        if not isValidInputTag(jetCollection) or jetCollection=="":
            print "INFO : jet collection %s does not exists, no energy resolution shifting will be performed in MET uncertainty tools" % jetCollection
        else:
            if addShiftedResJetCollections:
                variations = { "ResUp":-1., "ResDown":1.  }
                for var in variations.keys():
                    jetCollectionToKeep = \
                        addSmearedJets(process, cleanedJetCollection,
                                       [ "smeared", jetCollection, var ],
                                       jetSmearFileName,jetSmearHistogram,jetResolutions,
                                       varyByNsigmas, variations[ var ],
                                       shiftedParticleSequence, postfix)
                    jetCol={'jetCollection%s'%var:jetCollectionToKeep}
                    shiftedParticleCollections.update( jetCol )
                    collectionsToKeep.append( jetCollectionToKeep )

        #--------------------------------------------------------------------------------------------
        # produce collection of jets shifted up/down in energy
        #--------------------------------------------------------------------------------------------
        if not isValidInputTag(jetCollection) or jetCollection=="":
            print "INFO : jet collection %s does not exists, no energy shifting will be performed in MET uncertainty tools" % jetCollection
        else:
            shiftedJetsCollections, jetsCollectionsToKeep = addShiftedJetCollections(
                process,jetCollection,lastJetCollection,
                jetCorrLabelUpToL3, jetCorrLabelUpToL3Res,
                jecUncertaintyFile, jecUncertaintyTag,
                varyByNsigmas, shiftedParticleSequence,
                postfix)
            shiftedParticleCollections.update( shiftedJetsCollections )
            collectionsToKeep.extend( jetsCollectionsToKeep )

        #--------------------------------------------------------------------------------------------
        # produce collection of electrons shifted up/down in energy
        #--------------------------------------------------------------------------------------------
        if not isValidInputTag(electronCollection) or electronCollection=="":
            print "INFO : electron collection %s does not exists, no energy shifting will be performed in MET uncertainty tools" % electronCollection
        else:
            shiftedElectronsCollections, electronsCollectionsToKeep = addShiftedSingleParticleCollection(
                process, "electron", electronCollection,
                varyByNsigmas,shiftedParticleSequence,
                postfix)

            shiftedParticleCollections.update( shiftedElectronsCollections )
            collectionsToKeep.extend( electronsCollectionsToKeep )

        #--------------------------------------------------------------------------------------------
        # produce collection of (high Pt) photon candidates shifted up/down in energy
        #--------------------------------------------------------------------------------------------
        if not isValidInputTag(photonCollection) or photonCollection=="":
            print "INFO : photon collection %s does not exists, no energy shifting will be performed in MET uncertainty tools" % photonCollection
        else:
            shiftedPhotonsCollections, photonsCollectionsToKeep = addShiftedSingleParticleCollection(
                process, "photon", photonCollection,
                varyByNsigmas,shiftedParticleSequence,
                postfix)
            shiftedParticleCollections.update( shiftedPhotonsCollections )
            collectionsToKeep.extend( photonsCollectionsToKeep )

        #--------------------------------------------------------------------------------------------
        # produce collection of muons shifted up/down in energy/momentum
        #--------------------------------------------------------------------------------------------
        if not isValidInputTag(muonCollection) or muonCollection=="":
            print "INFO : muon collection %s does not exists, no energy shifting will be performed in MET uncertainty tools" % muonCollection
        else:
            shiftedMuonsCollections, muonsCollectionsToKeep = addShiftedSingleParticleCollection(process, "muon", muonCollection,
                                               varyByNsigmas,shiftedParticleSequence,
                                               postfix)
            shiftedParticleCollections.update( shiftedMuonsCollections )
            collectionsToKeep.extend( muonsCollectionsToKeep )

        #--------------------------------------------------------------------------------------------
        # produce collection of tau-jets shifted up/down in energy
        #--------------------------------------------------------------------------------------------
        if not isValidInputTag(tauCollection) or tauCollection=="":
            print "INFO : tau collection %s does not exists, no energy shifting  will be performed in MET uncertainty tools" % tauCollection
        else:
            shiftedTausCollections, tausCollectionsToKeep = addShiftedSingleParticleCollection(process, "tau", tauCollection,
                                               varyByNsigmas,shiftedParticleSequence,
                                               postfix)
            shiftedParticleCollections.update( shiftedTausCollections )
            collectionsToKeep.extend( tausCollectionsToKeep )


        return ( shiftedParticleSequence, shiftedParticleCollections, collectionsToKeep )





    def _addPFCandidatesForPFMEtInput(self, process, metUncertaintySequence,
                                      particleCollection, particleType, shiftType, particleCollectionShiftUp, particleCollectionShiftDown,
                                      dRmatch,
                                      pfCandCollection, postfix):

        srcUnshiftedObjects = particleCollection
        if isinstance(srcUnshiftedObjects, cms.InputTag):
            srcUnshiftedObjects = srcUnshiftedObjects.value()
        moduleShiftUp = cms.EDProducer("ShiftedPFCandidateProducerByMatchedObject",
            srcPFCandidates = pfCandCollection,
            srcUnshiftedObjects = cms.InputTag(srcUnshiftedObjects),
            srcShiftedObjects = cms.InputTag(particleCollectionShiftUp),
            dRmatch_PFCandidate = cms.double(dRmatch)
        )
        moduleNameShiftUp = "pfCandidates%s%sUp%s" % (particleType, shiftType, postfix)
        setattr(process, moduleNameShiftUp, moduleShiftUp)
        metUncertaintySequence += moduleShiftUp

        moduleShiftDown = moduleShiftUp.clone(
            srcShiftedObjects = cms.InputTag(particleCollectionShiftDown)
        )
        moduleNameShiftDown = "pfCandidates%s%sDown%s" % (particleType, shiftType, postfix)
        setattr(process, moduleNameShiftDown, moduleShiftDown)
        metUncertaintySequence += moduleShiftDown

        moduleNameShifts = { 'Up':moduleNameShiftUp , 'Down':moduleNameShiftDown }
        return moduleNameShifts
 #       return ( moduleNameShiftUp, moduleNameShiftDown )

    def _getLeptonsForPFMEtInput(self, shiftedParticleCollections, substituteKeyUnshifted = None, substituteKeyShifted = None, postfix=""):
        retVal = []
        for particleCol in shiftedParticleCollections.keys():
            if not (particleCol.find("Jet") or particleCol.find("jet") ) and isValidInputTag(shiftedParticleCollections[collectionName]):
                print collectionName,"==>:"+shiftedParticleCollections[collectionName]
                if substituteKeyUnshifted is not None and substituteKeyUnshifted in shiftedParticleCollections.keys() and \
                   substituteKeyShifted is not None and substituteKeyShifted in shiftedParticleCollections.keys() and \
                   shiftedParticleCollections[collectionName] == shiftedParticleCollections[substituteKeyUnshifted]:
                    retVal.append(cms.InputTag(shiftedParticleCollections[substituteKeyShifted]))
                else:
                    retVal.append(shiftedParticleCollections[collectionName])
        return retVal

    def _addPATMEtProducer(self, process, metUncertaintySequence,
                           metCollection, patMEtCollection,
                           collectionsToKeep, postfix):

        module = patMETs.clone(
            metSource = cms.InputTag(metCollection),
            addMuonCorrections = cms.bool(False),
            genMETSource = cms.InputTag('genMetTrue')
        )
        patMEtCollectionName = patMEtCollection+postfix
        setattr(process, patMEtCollectionName, module)
        metUncertaintySequence += module
        collectionsToKeep.append(patMEtCollectionName)


    def __call__(self, process,
                 electronCollection      = None,
                 photonCollection        = None,
                 muonCollection          = None,
                 tauCollection           = None,
                 jetCollection           = None,
                 jetCorrLabel            = None,
                 doSmearJets             = None,
                 jetSmearFileName        = None,
                 jetSmearHistogram       = None,
                 jetCorrPayloadName      = None,
                 jetCorrLabelUpToL3      = None,
                 jetCorrLabelUpToL3Res   = None,
                 jecUncertaintyFile      = None,
                 jecUncertaintyTag       = None,
                 varyByNsigmas           = None,
                 addToPatDefaultSequence = None,
                 outputModule            = None,
                 postfix                 = None):
        electronCollection = self._initializeInputTag(electronCollection, 'electronCollection')
        photonCollection = self._initializeInputTag(photonCollection, 'photonCollection')
        muonCollection = self._initializeInputTag(muonCollection, 'muonCollection')
        tauCollection = self._initializeInputTag(tauCollection, 'tauCollection')
        jetCollection = self._initializeInputTag(jetCollection, 'jetCollection')
        if jetCorrLabel is None:
            jetCorrLabel = self._defaultParameters['jetCorrLabel'].value
        if doSmearJets is None:
            doSmearJets = self._defaultParameters['doSmearJets'].value
        if jetSmearFileName is None:
            jetSmearFileName = self._defaultParameters['jetSmearFileName'].value
        if jetSmearHistogram is None:
            jetSmearHistogram = self._defaultParameters['jetSmearHistogram'].value
        if jetCorrPayloadName is None:
            jetCorrPayloadName = self._defaultParameters['jetCorrPayloadName'].value
        if jetCorrLabelUpToL3 is None:
            jetCorrLabelUpToL3 = self._defaultParameters['jetCorrLabelUpToL3'].value
        if jetCorrLabelUpToL3Res is None:
            jetCorrLabelUpToL3Res = self._defaultParameters['jetCorrLabelUpToL3Res'].value
        if jecUncertaintyFile is None:
            jecUncertaintyFile = self._defaultParameters['jecUncertaintyFile'].value
        if jecUncertaintyTag is None:
            jecUncertaintyTag = self._defaultParameters['jecUncertaintyTag'].value
        if varyByNsigmas is None:
            varyByNsigmas = self._defaultParameters['varyByNsigmas'].value
        if addToPatDefaultSequence is None:
            addToPatDefaultSequence = self._defaultParameters['addToPatDefaultSequence'].value
        if outputModule is None:
            outputModule = self._defaultParameters['outputModule'].value
        if postfix is None:
            postfix = self._defaultParameters['postfix'].value

        self.setParameter('electronCollection', electronCollection)
        self.setParameter('photonCollection', photonCollection)
        self.setParameter('muonCollection', muonCollection)
        self.setParameter('tauCollection', tauCollection)
        self.setParameter('jetCollection', jetCollection)
        self.setParameter('jetCorrLabel', jetCorrLabel)
        self.setParameter('doSmearJets', doSmearJets)
        self.setParameter('jetSmearFileName', jetSmearFileName)
        self.setParameter('jetSmearHistogram', jetSmearHistogram)
        self.setParameter('jetCorrPayloadName', jetCorrPayloadName)
        self.setParameter('jetCorrLabelUpToL3', jetCorrLabelUpToL3)
        self.setParameter('jetCorrLabelUpToL3Res', jetCorrLabelUpToL3Res)
        self.setParameter('jecUncertaintyFile', jecUncertaintyFile)
        self.setParameter('jecUncertaintyTag', jecUncertaintyTag)
        self.setParameter('varyByNsigmas', varyByNsigmas)
        self.setParameter('addToPatDefaultSequence', addToPatDefaultSequence)
        self.setParameter('outputModule', outputModule)
        self.setParameter('postfix', postfix)

