import FWCore.ParameterSet.Config as cms

from FWCore.GuiBrowsers.ConfigToolBase import *

import PhysicsTools.PatAlgos.tools.helpers as configtools
from PhysicsTools.PatAlgos.tools.trigTools import _addEventContent
from PhysicsTools.PatUtils.tools.jmeUncertaintyTools import JetMEtUncertaintyTools

from PhysicsTools.PatUtils.patPFMETCorrections_cff import *
import RecoMET.METProducers.METSigParams_cfi as jetResolutions
from PhysicsTools.PatAlgos.producersLayer1.metProducer_cfi import patMETs
 
class RunType1CaloMEtUncertainties(JetMEtUncertaintyTools):

    """ Shift energy of jets and "unclustered energy" reconstructed in the event up/down,
    in order to estimate effect of energy scale uncertainties on Type-1 corrected CaloMET
   """
    _label='runType1CaloMEtUncertainties'
    _defaultParameters = dicttypes.SortedKeysDict()
    def __init__(self):
        JetMEtUncertaintyTools.__init__(self)
	self.addParameter(self._defaultParameters, 'dRjetCleaning', 0.5, 
                          "Eta-phi distance for extra jet cleaning", Type=float)
	self.addParameter(self._defaultParameters, 'caloTowerCollection', cms.InputTag('towerMaker'), 
                          "Input CaloTower collection", Type=cms.InputTag)
        self.addParameter(self._defaultParameters, 'type1JetPtThreshold', 20.0, 
                          "Jet Pt threshold for Type-1 MET correction", Type=float)
        self._parameters = copy.deepcopy(self._defaultParameters)
        self._comment = ""
   
    def _addCorrCaloMEt(self, process, metUncertaintySequence,
                        shiftedParticleCollections, caloTowerCollection,
                        collectionsToKeep,
                        jetCorrLabelUpToL3, jetCorrLabelUpToL3Res,
                        jecUncertaintyFile, jecUncertaintyTag,
                        varyByNsigmas,
                        type1JetPtThreshold,
                        postfix):

        ## standard naming convention
        metModNameT1="patCaloMetT1"
        metModNameT1T2="patCaloMetT1T2"

        if not hasattr(process, 'produceCaloMETCorrectors'):
            process.load("JetMETCorrections.Type1MET.caloMETCorrections_cff")
        process.caloJetMETcorr.type1JetPtThreshold = cms.double(type1JetPtThreshold)

        #rename modules
        process.caloMetT1 = process.caloType1CorrectedMet.clone()
        process.caloMetT1T2 = process.caloType1CorrectedMet.clone()
        process.produceCaloMETCorrectors += process.caloMetT1
        process.produceCaloMETCorrectors += process.caloMetT1T2

        # If with empty postfix, make a backup of
        # process.producePatPFMETCorrections, because the original
        # sequence will be modified later in this function
        if postfix == "":
            configtools.cloneProcessingSnippet(process, process.produceCaloMETCorrectors, "OriginalReserved")
        else:
            if postfix == "OriginalReserved":
                raise ValueError("Postfix label '%s' is reserved for internal usage !!" % postfix)

            if hasattr(process, "produceCaloMETCorrectorsOriginalReserved"):
                configtools.cloneProcessingSnippet(process, process.produceCaloMETCorrectorsOriginalReserved, postfix, removePostfix="OriginalReserved")
            else:
                configtools.cloneProcessingSnippet(process, process.produceCaloMETCorrectors, postfix)

        metUncertaintySequence += getattr(process, "produceCaloMETCorrectors")

        metCollectionsUp_DownForCorrMEt = \
            self._propagateMEtUncertainties(
              process, shiftedParticleCollections['lastJetCollection'], "Jet", "En",
              shiftedParticleCollections['jetCollectionEnUpForCorrMEt'], shiftedParticleCollections['jetCollectionEnDownForCorrMEt'],
              process.caloMetT1, "Calo", metUncertaintySequence, postfix)
        #getattr(process, "patCaloMETcorrJetEnUp").verbosity = cms.int32(1)
        #getattr(process, metCollectionsUp_DownForCorrMEt[0]).verbosity = cms.int32(1)
        #getattr(process, "patCaloMETcorrJetEnDown").verbosity = cms.int32(1)
        #getattr(process, metCollectionsUp_DownForCorrMEt[1]).verbosity = cms.int32(1)X

        self._addPATMEtProducer(process, metUncertaintySequence,
                                metCollectionsUp_DownForCorrMEt[0], 'patCaloMetT1JetEnUp', collectionsToKeep, postfix)
        self._addPATMEtProducer(process, metUncertaintySequence,
                                metCollectionsUp_DownForCorrMEt[1], 'patCaloMetT1JetEnDown', collectionsToKeep, postfix)
        
        if caloTowerCollection.value() != "\"\"":
            process.caloJetMETcorr.srcMET = cms.InputTag('')
            process.caloTowersNotInJetsForMEtUncertainty = cms.EDProducer("TPCaloJetsOnCaloTowers",
                enable = cms.bool(True),
                verbose = cms.untracked.bool(False),
                name = cms.untracked.string("caloTowersNotInJetsForMEtUncertainty"),
                topCollection = cms.InputTag('ak5CaloJets'),
                bottomCollection = caloTowerCollection
            )
            metUncertaintySequence += process.caloTowersNotInJetsForMEtUncertainty
            process.caloTowerMETcorr = cms.EDProducer("CaloTowerMETcorrInputProducer",
                src = cms.InputTag('caloTowersNotInJetsForMEtUncertainty'),
                residualCorrLabel = cms.string(""),
                residualCorrEtaMax = cms.double(9.9),
                residualCorrOffset = cms.double(0.),
                isMC = cms.bool(False), # CV: only used to decide whether to apply "unclustered energy" calibration to MC or Data
                globalThreshold = cms.double(0.3), # NOTE: this value need to match met.globalThreshold, defined in RecoMET/METProducers/python/CaloMET_cfi.py
                noHF = cms.bool(False),
                #verbosity = cms.int32(1)
            )
            metUncertaintySequence += process.caloTowerMETcorr
            moduleCaloType1CorrectedMetUnclusteredEnUp = getattr(process, "patCaloMetT1" + postfix).clone(
                applyType2Corrections = cms.bool(True),
                srcUnclEnergySums = cms.VInputTag(
                    cms.InputTag('caloJetMETcorr', 'type2'),
                    cms.InputTag('caloTowerMETcorr')
                ),
                type2CorrFormula = cms.string("A"),
                type2CorrParameter = cms.PSet(
                    A = cms.double(1.0 + 0.1*varyByNsigmas)
                ),
                #verbosity = cms.int32(1)
            )
            moduleCaloType1CorrectedMetUnclusteredEnUpName = "patCaloMetT1UnclusteredEnUp%s" % postfix
            setattr(process, moduleCaloType1CorrectedMetUnclusteredEnUpName, moduleCaloType1CorrectedMetUnclusteredEnUp)
            metUncertaintySequence += getattr(process, moduleCaloType1CorrectedMetUnclusteredEnUpName)
            self._addPATMEtProducer(process, metUncertaintySequence,
                                    moduleCaloType1CorrectedMetUnclusteredEnUpName, 'patCaloMetT1UnclusteredEnUp', collectionsToKeep, postfix)
            moduleCaloType1CorrectedMetUnclusteredEnDown = moduleCaloType1CorrectedMetUnclusteredEnUp.clone(
                type2CorrParameter = cms.PSet(
                    A = cms.double(1.0 - 0.1*varyByNsigmas)
                )
            )
            moduleCaloType1CorrectedMetUnclusteredEnDownName = "patCaloMetT1UnclusteredEnDown%s" % postfix
            setattr(process, moduleCaloType1CorrectedMetUnclusteredEnDownName, moduleCaloType1CorrectedMetUnclusteredEnDown)
            metUncertaintySequence += getattr(process, moduleCaloType1CorrectedMetUnclusteredEnDownName)
            self._addPATMEtProducer(process, metUncertaintySequence,
                                    moduleCaloType1CorrectedMetUnclusteredEnDownName, 'patCaloMetT1UnclusteredEnDown', collectionsToKeep, postfix)
        else:
            moduleCaloType1CorrectedMetUnclusteredEnUp = getattr(process, "caloMetT1" + postfix).clone(
                applyType2Corrections = cms.bool(True),
                srcUnclEnergySums = cms.VInputTag(
                    cms.InputTag('caloJetMETcorr', 'type2fromMEt')
                ),
                type2CorrFormula = cms.string("A"),
                type2CorrParameter = cms.PSet(
                    A = cms.double(1.0 + 0.1*varyByNsigmas)
                )
            )
            moduleCaloType1CorrectedMetUnclusteredEnUpName = "caloMetT1UnclusteredEnUp%s" % postfix
            setattr(process, moduleCaloType1CorrectedMetUnclusteredEnUpName, moduleCaloType1CorrectedMetUnclusteredEnUp)
            metUncertaintySequence += getattr(process, moduleCaloType1CorrectedMetUnclusteredEnUpName)
            self._addPATMEtProducer(process, metUncertaintySequence,
                                    moduleCaloType1CorrectedMetUnclusteredEnUpName, 'patCaloMetT1UnclusteredEnUp', collectionsToKeep, postfix)
            moduleCaloType1CorrectedMetUnclusteredEnDown = moduleCaloType1CorrectedMetUnclusteredEnUp.clone(
                type2CorrParameter = cms.PSet(
                    A = cms.double(1.0 - 0.1*varyByNsigmas)
                )
            )
            moduleCaloType1CorrectedMetUnclusteredEnDownName = "caloMetT1UnclusteredEnDown%s" % postfix
            setattr(process, moduleCaloType1CorrectedMetUnclusteredEnDownName, moduleCaloType1CorrectedMetUnclusteredEnDown)
            metUncertaintySequence += getattr(process, moduleCaloType1CorrectedMetUnclusteredEnDownName)
            self._addPATMEtProducer(process, metUncertaintySequence,
                                    moduleCaloType1CorrectedMetUnclusteredEnDownName, 'patCaloMetT1UnclusteredEnDown', collectionsToKeep, postfix)
            
    def __call__(self, process,
                 electronCollection      = None,
                 photonCollection        = None,
                 muonCollection          = None,
                 tauCollection           = None,
                 jetCollection           = None,
                 dRjetCleaning           = None,
                 caloTowerCollection     = None,
                 jetCorrPayloadName      = None,
                 jetCorrLabelUpToL3      = None,
                 jetCorrLabelUpToL3Res   = None,
                 jecUncertaintyFile      = None,
                 jecUncertaintyTag       = None,
                 varyByNsigmas           = None,
                 type1JetPtThreshold     = None,
                 addToPatDefaultSequence = None,
                 outputModule            = None,
                 postfix                 = None):
        JetMEtUncertaintyTools.__call__(
            self, process,
            electronCollection = electronCollection,
            photonCollection = photonCollection,
            muonCollection = muonCollection,
            tauCollection = tauCollection,
            jetCollection = jetCollection,
            jetCorrLabel = None,
            doSmearJets = False,
            jetCorrPayloadName = jetCorrPayloadName,
            jetCorrLabelUpToL3 = jetCorrLabelUpToL3,
            jetCorrLabelUpToL3Res = jetCorrLabelUpToL3Res,
            jecUncertaintyFile = jecUncertaintyFile,
            jecUncertaintyTag = jecUncertaintyTag,
            varyByNsigmas = varyByNsigmas,
            addToPatDefaultSequence = addToPatDefaultSequence,
            outputModule = outputModule,
            postfix = postfix)
        if dRjetCleaning is None:
            dRjetCleaning = self._defaultParameters['dRjetCleaning'].value
        caloTowerCollection = self._initializeInputTag(caloTowerCollection, 'caloTowerCollection')
        if type1JetPtThreshold is None:
            type1JetPtThreshold = self._defaultParameters['type1JetPtThreshold'].value

        self.setParameter('dRjetCleaning', dRjetCleaning)
        self.setParameter('caloTowerCollection', caloTowerCollection)
        self.setParameter('type1JetPtThreshold', type1JetPtThreshold)
  
        self.apply(process) 
        
    def toolCode(self, process):        
        electronCollection = self._parameters['electronCollection'].value
        photonCollection = self._parameters['photonCollection'].value
        muonCollection = self._parameters['muonCollection'].value
        tauCollection = self._parameters['tauCollection'].value
        jetCollection = self._parameters['jetCollection'].value
        jetCorrLabel = self._parameters['jetCorrLabel'].value
        dRjetCleaning =  self._parameters['dRjetCleaning'].value
        caloTowerCollection = self._parameters['caloTowerCollection'].value
        jetCorrPayloadName = self._parameters['jetCorrPayloadName'].value
        jetCorrLabelUpToL3 = self._parameters['jetCorrLabelUpToL3'].value
        jetCorrLabelUpToL3Res = self._parameters['jetCorrLabelUpToL3Res'].value
        jecUncertaintyFile = self._parameters['jecUncertaintyFile'].value
        jecUncertaintyTag = self._parameters['jecUncertaintyTag'].value
        varyByNsigmas = self._parameters['varyByNsigmas'].value
        type1JetPtThreshold = self._parameters['type1JetPtThreshold'].value
        addToPatDefaultSequence = self._parameters['addToPatDefaultSequence'].value
        outputModule = self._parameters['outputModule'].value
        postfix = self._parameters['postfix'].value

        if not hasattr(process, "caloType1MEtUncertaintySequence" + postfix):
            metUncertaintySequence = cms.Sequence()
            setattr(process, "caloType1MEtUncertaintySequence" + postfix, metUncertaintySequence)
        metUncertaintySequence = getattr(process, "caloType1MEtUncertaintySequence" + postfix)

        collectionsToKeep = []

        # produce collection of jets not overlapping with reconstructed
        # electrons/photons, muons and tau-jet candidates
        lastJetCollection, cleanedJetCollection = \
            self._addCleanedJets(process, jetCollection,
                                 electronCollection, photonCollection, muonCollection, tauCollection,
                                 metUncertaintySequence, postfix)

        moduleJetsNotOverlappingWithLeptonsPtGtType1Threshold = cms.EDFilter("PATJetSelector",
            src = cms.InputTag(lastJetCollection),
            cut = cms.string('pt > %f' % type1JetPtThreshold)
        )
        moduleJetsNotOverlappingWithLeptonsPtGtType1ThresholdName = "%sNotOverlappingWithLeptonsPtGtType1ThresholdForJetMEtUncertainty" % jetCollection.value()
        setattr(process, moduleJetsNotOverlappingWithLeptonsPtGtType1ThresholdName, moduleJetsNotOverlappingWithLeptonsPtGtType1Threshold)
        metUncertaintySequence += moduleJetsNotOverlappingWithLeptonsPtGtType1Threshold
        lastJetCollection = moduleJetsNotOverlappingWithLeptonsPtGtType1ThresholdName
        
        collectionsToKeep.append(lastJetCollection)

        #--------------------------------------------------------------------------------------------    
        # produce collection of electrons/photons, muons, tau-jet candidates and jets
        # shifted up/down in energy by their respective energy uncertainties
        #--------------------------------------------------------------------------------------------

        shiftedParticleSequence, shiftedParticleCollections, addCollectionsToKeep = \
          self._addShiftedParticleCollections(process,
                                              electronCollection,
                                              photonCollection,
                                              muonCollection,
                                              tauCollection,
                                              jetCollection, cleanedJetCollection, lastJetCollection,
                                              None, None,
                                              jetCorrLabelUpToL3, jetCorrLabelUpToL3Res,
                                              jecUncertaintyFile, jecUncertaintyTag,
                                              varyByNsigmas,
                                              postfix)
        setattr(process, "shiftedParticlesForType1CaloMEtUncertainties" + postfix, shiftedParticleSequence)        
        metUncertaintySequence += getattr(process, "shiftedParticlesForType1CaloMEtUncertainties" + postfix)
        collectionsToKeep.extend(addCollectionsToKeep)
        
        #--------------------------------------------------------------------------------------------    
        # propagate shifted particle energies to Type 1 and Type 1 + 2 corrected PFMET
        #--------------------------------------------------------------------------------------------

        self._addCorrCaloMEt(process, metUncertaintySequence,
                             shiftedParticleCollections, caloTowerCollection,
                             collectionsToKeep,
                             jetCorrLabelUpToL3, jetCorrLabelUpToL3Res,
                             jecUncertaintyFile, jecUncertaintyTag,
                             varyByNsigmas,
                             type1JetPtThreshold,
                             postfix)
        
        # insert metUncertaintySequence into patDefaultSequence
        if addToPatDefaultSequence:
            if not hasattr(process, "patDefaultSequence"):
                raise ValueError("PAT default sequence is not defined !!")
            process.patDefaultSequence += metUncertaintySequence        
       
        # add shifted + unshifted collections pf pat::Electrons/Photons,
        # Muons, Taus, Jets and MET to PAT-tuple event content
        if outputModule is not None and hasattr(process, outputModule):
            getattr(process, outputModule).outputCommands = _addEventContent(
                getattr(process, outputModule).outputCommands,
                [ 'keep *_%s_*_%s' % (collectionToKeep, process.name_()) for collectionToKeep in collectionsToKeep ])
       
runType1CaloMEtUncertainties = RunType1CaloMEtUncertainties()
