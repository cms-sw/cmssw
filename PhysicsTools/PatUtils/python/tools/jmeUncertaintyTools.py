import FWCore.ParameterSet.Config as cms

from FWCore.GuiBrowsers.ConfigToolBase import *

import PhysicsTools.PatAlgos.tools.helpers as configtools
from PhysicsTools.PatAlgos.tools.trigTools import _addEventContent

from PhysicsTools.PatUtils.patPFMETCorrections_cff import *
import RecoMET.METProducers.METSigParams_cfi as jetResolutions
from PhysicsTools.PatAlgos.producersLayer1.metProducer_cfi import patMETs
 
##MM
from PhysicsTools.PatAlgos.patSequences_cff import *

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
        self.addParameter(self._defaultParameters, 'jetCorrLabel', "L3Absolute", 
                          "NOTE: use 'L3Absolute' for MC/'L2L3Residual' for Data", Type=str)
	self.addParameter(self._defaultParameters, 'doSmearJets', True, 
                          "Flag to enable/disable jet smearing to better match MC to Data", Type=bool)
	self.addParameter(self._defaultParameters, 'jetSmearFileName', 'PhysicsTools/PatUtils/data/pfJetResolutionMCtoDataCorrLUT.root', 
                          "Name of ROOT file containing histogram with jet smearing factors", Type=str) 
        self.addParameter(self._defaultParameters, 'jetSmearHistogram', 'pfJetResolutionMCtoDataCorrLUT', 
                          "Name of histogram with jet smearing factors", Type=str) 
	self.addParameter(self._defaultParameters, 'jetCorrPayloadName', 'AK5PF', 
                          "Use AK5PF for PFJets, AK5Calo for CaloJets", Type=str)
        self.addParameter(self._defaultParameters, 'jetCorrLabelUpToL3', 'ak5PFL1FastL2L3',
                          "Use ak5PFL1FastL2L3 (ak5PFchsL1FastL2L3) for PFJets with (without) charged hadron subtraction, ak5CaloL1FastL2L3 for CaloJets", Type=str)
        self.addParameter(self._defaultParameters, 'jetCorrLabelUpToL3Res', 'ak5PFL1FastL2L3Residual',
                          "Use ak5PFL1FastL2L3Residual (ak5PFchsL1FastL2L3Residual) for PFJets with (without) charged hadron subtraction, ak5CaloL1FastL2L3Residual for CaloJets", Type=str)
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

    def _addModuleToSequence(self, process, module, moduleName_parts, sequence, postfix):

        if not len(moduleName_parts) > 0:
            raise ValueError("Empty list !!")

        moduleName = ""

        lastPart = None
        for part in moduleName_parts:
            if part is None or part == "":
                continue

            part = part.replace("selected", "")
            part = part.replace("clean",    "")

            if lastPart is None:
                moduleName += part[0].lower() + part[1:]
                lastPart = part
            else:
                if lastPart[-1].islower() or lastPart[-1].isdigit():
                    moduleName += part[0].capitalize() + part[1:]
                else:
                    moduleName += part[0].lower() + part[1:]
                lastPart = part    

        moduleName += postfix
        setattr(process, moduleName, module)

        sequence += module
 
        return moduleName

    def _addCleanedJets(self, process, jetCollection,
                        electronCollection, photonCollection, muonCollection, tauCollection,
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
        for collection in [
            [ 'electrons', electronCollection ],
            [ 'photons',   photonCollection   ],
            [ 'muons',     muonCollection     ],
            [ 'taus',      tauCollection      ] ]:
            if self._isValidInputTag(collection[1]):
                setattr(jetsNotOverlappingWithLeptons.checkOverlaps, collection[0], cms.PSet(
                    src                 = collection[1],
                    algorithm           = cms.string("byDeltaR"),
                    preselection        = cms.string(""),
                    deltaR              = cms.double(0.5),
                    checkRecoComponents = cms.bool(False), 
                    pairCut             = cms.string(""),
                    requireNoOverlaps   = cms.bool(True),
                ))
                numOverlapCollections = numOverlapCollections + 1
        lastJetCollection = jetCollection.value()        
        if numOverlapCollections >= 1:
            lastJetCollection = \
              self._addModuleToSequence(process, jetsNotOverlappingWithLeptons,
                                        [ jetCollection.value(), "NotOverlappingWithLeptonsForJetMEtUncertainty" ],
                                        uncertaintySequence, postfix)
        cleanedJetCollection = lastJetCollection

        return ( lastJetCollection, cleanedJetCollection )

    def _addSmearedJets(self, process, jetCollection, smearedJetCollectionName_parts,
                        jetSmearFileName, jetSmearHistogram, varyByNsigmas,
                        shiftBy = None,
                        uncertaintySequence = None, postfix = ""):

        smearedJets = cms.EDProducer("SmearedPATJetProducer",
            src = cms.InputTag(jetCollection),
            dRmaxGenJetMatch = cms.string('TMath::Min(0.5, 0.1 + 0.3*TMath::Exp(-0.05*(genJetPt - 10.)))'),
            sigmaMaxGenJetMatch = cms.double(3.),                               
            inputFileName = cms.FileInPath(jetSmearFileName),
            lutName = cms.string(jetSmearHistogram),
            jetResolutions = jetResolutions.METSignificance_params,
            # CV: skip jet smearing for pat::Jets for which the jet-energy correction (JEC) factors are either very large or negative
            #     since both cases produce unphysically large tails in the Type 1 corrected MET distribution after the smearing,
            #
            #     e.g. raw jet:   energy = 50 GeV, eta = 2.86, pt =  1   GeV 
            #          corr. jet: energy = -3 GeV            , pt = -0.1 GeV (JEC factor L1fastjet*L2*L3 = -17)
            #                     energy = 10 GeV for corrected jet after smearing
            #         --> smeared raw jet energy = -170 GeV !!
            #
            #         --> (corr. - raw) jet contribution to MET = -1 (-10) GeV before (after) smearing,
            #             even though jet energy got smeared by merely 1 GeV
            #
            skipJetSelection = cms.string(
                'jecSetsAvailable & abs(energy - correctedP4("Uncorrected").energy) > (5.*min(energy, correctedP4("Uncorrected").energy))'
            ),
            skipRawJetPtThreshold = cms.double(10.), # GeV
            skipCorrJetPtThreshold = cms.double(1.e-2),
            verbosity = cms.int32(0)                                     
        )
        if shiftBy is not None:
            setattr(smearedJets, "shiftBy", cms.double(shiftBy*varyByNsigmas))
        smearedJetCollection = \
          self._addModuleToSequence(process, smearedJets,
                                    smearedJetCollectionName_parts,
                                    uncertaintySequence, postfix)

        return smearedJetCollection
        
    def _propagateMEtUncertainties(self, process,
                                   particleCollection, particleType, shiftType, particleCollectionShiftUp, particleCollectionShiftDown,
                                   metProducer, metType, sequence, postfix):

        # produce MET correction objects
        # (sum of differences in four-momentum between original and up/down shifted particle collection)
        moduleMETcorrShiftUp = cms.EDProducer("ShiftedParticleMETcorrInputProducer",
            srcOriginal = cms.InputTag(particleCollection),
            srcShifted = cms.InputTag(particleCollectionShiftUp)                                                          
        )
        moduleMETcorrShiftUpName = "pat%sMETcorr%s%sUp%s" % (metType, particleType, shiftType, postfix)
        setattr(process, moduleMETcorrShiftUpName, moduleMETcorrShiftUp)
        sequence += moduleMETcorrShiftUp
        moduleMETcorrShiftDown = moduleMETcorrShiftUp.clone(
            srcShifted = cms.InputTag(particleCollectionShiftDown)                                           
        )
        moduleMETcorrShiftDownName = "pat%sMETcorr%s%sDown%s" % (metType, particleType, shiftType, postfix)
        setattr(process, moduleMETcorrShiftDownName, moduleMETcorrShiftDown)
        sequence += moduleMETcorrShiftDown

        # propagate effects of up/down shifts to MET
        moduleMETshiftUp = metProducer.clone(
            src = cms.InputTag(metProducer.label()),
            srcType1Corrections = cms.VInputTag(
                cms.InputTag(moduleMETcorrShiftUpName)
            ),
            srcUnclEnergySums = cms.VInputTag(),
            applyType2Corrections = cms.bool(False),
            type2CorrParameter = cms.PSet(
                A = cms.double(1.0)
            )
        )
        metProducerLabel = metProducer.label()
        if postfix != "":
            if metProducerLabel[-len(postfix):] == postfix:
                metProducerLabel = metProducerLabel[0:-len(postfix)]
            else:
                raise StandardError("Tried to remove postfix %s from label %s, but it wasn't there" % (postfix, metProducerLabel))
        moduleMETshiftUpName = "%s%s%sUp%s" % (metProducerLabel, particleType, shiftType, postfix)
        setattr(process, moduleMETshiftUpName, moduleMETshiftUp)
        sequence += moduleMETshiftUp
        moduleMETshiftDown = moduleMETshiftUp.clone(
            srcType1Corrections = cms.VInputTag(
                cms.InputTag(moduleMETcorrShiftDownName)
            )
        )
        moduleMETshiftDownName = "%s%s%sDown%s" % (metProducerLabel, particleType, shiftType, postfix)
        setattr(process, moduleMETshiftDownName, moduleMETshiftDown)
        sequence += moduleMETshiftDown

        metCollectionsUp_Down = [
            moduleMETshiftUpName,
            moduleMETshiftDownName
        ]

        return metCollectionsUp_Down

    def _initializeInputTag(self, input, default):
        retVal = None
        if input is None:
            retVal = self._defaultParameters[default].value
        elif type(input) == str:
            retVal = cms.InputTag(input)
        else:
            retVal = input
        return retVal

    @staticmethod
    def _isValidInputTag(input):
        input_str = input
        if isinstance(input, cms.InputTag):
            input_str = input.value()
        if input is None or input_str == '""':
            return False
        else:
            return True

    def _addShiftedParticleCollections(self, process, 
                                       electronCollection = None,
                                       photonCollection = None,
                                       muonCollection = None,
                                       tauCollection = None,
                                       jetCollection = None, cleanedJetCollection = None, lastJetCollection = None,
                                       jetCollectionResUp = None, jetCollectionResDown = None,
                                       jetCorrLabelUpToL3 = None, jetCorrLabelUpToL3Res = None,
                                       jecUncertaintyFile = None, jecUncertaintyTag = None,
                                       varyByNsigmas = None,
                                       postfix = ""):

        shiftedParticleSequence = cms.Sequence()
        
        shiftedParticleCollections = {}        
        shiftedParticleCollections['electronCollection'] = electronCollection
        shiftedParticleCollections['photonCollection'] = photonCollection
        shiftedParticleCollections['muonCollection'] = muonCollection
        shiftedParticleCollections['tauCollection'] = tauCollection
        shiftedParticleCollections['jetCollection'] = jetCollection
        shiftedParticleCollections['cleanedJetCollection'] = cleanedJetCollection
        shiftedParticleCollections['lastJetCollection'] = lastJetCollection
        shiftedParticleCollections['jetCollectionResUp'] = jetCollectionResUp
        shiftedParticleCollections['jetCollectionResDown'] = jetCollectionResDown
        collectionsToKeep = []
        
        #--------------------------------------------------------------------------------------------
        # produce collection of jets shifted up/down in energy    
        #--------------------------------------------------------------------------------------------     

        # in case of "raw" (uncorrected) MET,
        # add residual jet energy corrections in quadrature to jet energy uncertainties:
        # cf. https://twiki.cern.ch/twiki/bin/view/CMS/MissingETUncertaintyPrescription        
        jetsEnUpForRawMEt = cms.EDProducer("ShiftedPATJetProducer",
            src = cms.InputTag(lastJetCollection),
            #jetCorrPayloadName = cms.string(jetCorrPayloadName),
            #jetCorrUncertaintyTag = cms.string('Uncertainty'),
            jetCorrInputFileName = cms.FileInPath(jecUncertaintyFile),
            jetCorrUncertaintyTag = cms.string(jecUncertaintyTag),
            addResidualJES = cms.bool(True),
            jetCorrLabelUpToL3 = cms.string(jetCorrLabelUpToL3),
            jetCorrLabelUpToL3Res = cms.string(jetCorrLabelUpToL3Res),
            shiftBy = cms.double(+1.*varyByNsigmas)
        )
        jetCollectionEnUpForRawMEt = \
          self._addModuleToSequence(process, jetsEnUpForRawMEt,
                                    [ "shifted", jetCollection.value(), "EnUpForRawMEt" ],
                                    shiftedParticleSequence, postfix)
        shiftedParticleCollections['jetCollectionEnUpForRawMEt'] = jetCollectionEnUpForRawMEt
        collectionsToKeep.append(jetCollectionEnUpForRawMEt)
        jetsEnDownForRawMEt = jetsEnUpForRawMEt.clone(
            shiftBy = cms.double(-1.*varyByNsigmas)
        )
        jetCollectionEnDownForRawMEt = \
          self._addModuleToSequence(process, jetsEnDownForRawMEt,
                                    [ "shifted", jetCollection.value(), "EnDownForRawMEt" ],
                                    shiftedParticleSequence, postfix)
        shiftedParticleCollections['jetCollectionEnDownForRawMEt'] = jetCollectionEnDownForRawMEt
        collectionsToKeep.append(jetCollectionEnDownForRawMEt)

        jetsEnUpForCorrMEt = jetsEnUpForRawMEt.clone(
            addResidualJES = cms.bool(False)
        )
        jetCollectionEnUpForCorrMEt = \
          self._addModuleToSequence(process, jetsEnUpForCorrMEt,
                                    [ "shifted", jetCollection.value(), "EnUpForCorrMEt" ],
                                    shiftedParticleSequence, postfix)
        shiftedParticleCollections['jetCollectionEnUpForCorrMEt'] = jetCollectionEnUpForCorrMEt
        collectionsToKeep.append(jetCollectionEnUpForCorrMEt)
        jetsEnDownForCorrMEt = jetsEnUpForCorrMEt.clone(
            shiftBy = cms.double(-1.*varyByNsigmas)
        )
        jetCollectionEnDownForCorrMEt = \
          self._addModuleToSequence(process, jetsEnDownForCorrMEt,
                                    [ "shifted", jetCollection.value(), "EnDownForCorrMEt" ],
                                    shiftedParticleSequence, postfix)
        shiftedParticleCollections['jetCollectionEnDownForCorrMEt'] = jetCollectionEnDownForCorrMEt
        collectionsToKeep.append(jetCollectionEnDownForCorrMEt)

        #--------------------------------------------------------------------------------------------
        # produce collection of electrons shifted up/down in energy
        #--------------------------------------------------------------------------------------------

        electronCollectionEnUp = None
        electronCollectionEnDown = None
        if self._isValidInputTag(electronCollection):
            electronsEnUp = cms.EDProducer("ShiftedPATElectronProducer",
                src = electronCollection,
                binning = cms.VPSet(
                    cms.PSet(
                        binSelection = cms.string('isEB'),
                        binUncertainty = cms.double(0.006)
                    ),
                    cms.PSet(
                        binSelection = cms.string('!isEB'),
                        binUncertainty = cms.double(0.015)
                    ),
                ),      
                shiftBy = cms.double(+1.*varyByNsigmas)
            )
            electronCollectionEnUp = \
              self._addModuleToSequence(process, electronsEnUp,
                                        [ "shifted", electronCollection.value(), "EnUp" ],
                                        shiftedParticleSequence, postfix)
            shiftedParticleCollections['electronCollectionEnUp'] = electronCollectionEnUp
            collectionsToKeep.append(electronCollectionEnUp)
            electronsEnDown = electronsEnUp.clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            )
            electronCollectionEnDown = \
              self._addModuleToSequence(process, electronsEnDown,
                                        [ "shifted", electronCollection.value(), "EnDown" ],
                                        shiftedParticleSequence, postfix)
            shiftedParticleCollections['electronCollectionEnDown'] = electronCollectionEnDown
            collectionsToKeep.append(electronCollectionEnDown)

        #--------------------------------------------------------------------------------------------
        # produce collection of (high Pt) photon candidates shifted up/down in energy
        #--------------------------------------------------------------------------------------------    

        photonCollectionEnUp = None
        photonCollectionEnDown = None    
        if self._isValidInputTag(photonCollection):
            photonsEnUp = cms.EDProducer("ShiftedPATPhotonProducer",
                src = photonCollection,
                binning = cms.VPSet(
                    cms.PSet(
                        binSelection = cms.string('isEB'),
                        binUncertainty = cms.double(0.01)
                    ),
                    cms.PSet(
                        binSelection = cms.string('!isEB'),
                        binUncertainty = cms.double(0.025)
                    ),
                ),                         
                shiftBy = cms.double(+1.*varyByNsigmas)
            )
            photonCollectionEnUp = \
              self._addModuleToSequence(process, photonsEnUp,
                                        [ "shifted", photonCollection.value(), "EnUp" ],
                                        shiftedParticleSequence, postfix)
            shiftedParticleCollections['photonCollectionEnUp'] = photonCollectionEnUp
            collectionsToKeep.append(photonCollectionEnUp)
            photonsEnDown = photonsEnUp.clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            )
            photonCollectionEnDown = \
              self._addModuleToSequence(process, photonsEnDown,
                                        [ "shifted", photonCollection.value(), "EnDown" ],
                                        shiftedParticleSequence, postfix)
            shiftedParticleCollections['photonCollectionEnDown'] = photonCollectionEnDown
            collectionsToKeep.append(photonCollectionEnDown)

        #--------------------------------------------------------------------------------------------
        # produce collection of muons shifted up/down in energy/momentum  
        #--------------------------------------------------------------------------------------------

        muonCollectionEnUp = None
        muonCollectionEnDown = None   
        if self._isValidInputTag(muonCollection):
            muonsEnUp = cms.EDProducer("ShiftedPATMuonProducer",
                src = muonCollection,
             #   uncertainty = cms.double(0.002),
                shiftBy = cms.double(+1.*varyByNsigmas),
                binning = cms.VPSet(
                    cms.PSet(
                        binSelection = cms.string('pt < 100'),
                        binUncertainty = cms.double(0.002)
                        ),
                    cms.PSet(
                        binSelection = cms.string('pt >= 100'),
                        binUncertainty = cms.double(0.05)
                    ),
                ),
            )
            muonCollectionEnUp = \
              self._addModuleToSequence(process, muonsEnUp,
                                        [ "shifted", muonCollection.value(), "EnUp" ],
                                        shiftedParticleSequence, postfix)
            shiftedParticleCollections['muonCollectionEnUp'] = muonCollectionEnUp
            collectionsToKeep.append(muonCollectionEnUp)
            muonsEnDown = muonsEnUp.clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            )
            muonCollectionEnDown = \
              self._addModuleToSequence(process, muonsEnDown,
                                        [ "shifted", muonCollection.value(), "EnDown" ],
                                        shiftedParticleSequence, postfix)
            shiftedParticleCollections['muonCollectionEnDown'] = muonCollectionEnDown
            collectionsToKeep.append(muonCollectionEnDown)

        #--------------------------------------------------------------------------------------------
        # produce collection of tau-jets shifted up/down in energy
        #--------------------------------------------------------------------------------------------     

        tauCollectionEnUp = None
        tauCollectionEnDown = None 
        if self._isValidInputTag(tauCollection):
            tausEnUp = cms.EDProducer("ShiftedPATTauProducer",
                src = tauCollection,
                uncertainty = cms.double(0.03),                      
                shiftBy = cms.double(+1.*varyByNsigmas)
            )
            tauCollectionEnUp = \
              self._addModuleToSequence(process, tausEnUp,
                                        [ "shifted", tauCollection.value(), "EnUp" ],
                                        shiftedParticleSequence, postfix)
            shiftedParticleCollections['tauCollectionEnUp'] = tauCollectionEnUp
            collectionsToKeep.append(tauCollectionEnUp)
            tausEnDown = tausEnUp.clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            )
            tauCollectionEnDown = \
              self._addModuleToSequence(process, tausEnDown,
                                        [ "shifted", tauCollection.value(), "EnDown" ],
                                        shiftedParticleSequence, postfix)
            shiftedParticleCollections['tauCollectionEnDown'] = tauCollectionEnDown
            collectionsToKeep.append(tauCollectionEnDown)

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

        return ( moduleNameShiftUp, moduleNameShiftDown )

    def _getLeptonsForPFMEtInput(self, shiftedParticleCollections, substituteKeyUnshifted = None, substituteKeyShifted = None, postfix=""):
        retVal = []
        for collectionName in [ 'electronCollection',
                                'photonCollection',
                                'muonCollection',
                                'tauCollection' ]:
            if self._isValidInputTag(shiftedParticleCollections[collectionName]):
                if substituteKeyUnshifted is not None and substituteKeyUnshifted in shiftedParticleCollections.keys() and \
                   substituteKeyShifted is not None and substituteKeyShifted in shiftedParticleCollections.keys() and \
                   shiftedParticleCollections[collectionName] == shiftedParticleCollections[substituteKeyUnshifted]:
                    retVal.append(cms.InputTag(shiftedParticleCollections[substituteKeyShifted]))
                else:
                    retVal.append(shiftedParticleCollections[collectionName])
        return retVal
            
    def _addPATMEtProducer(self, process, metUncertaintySequence,
                           pfMEtCollection, patMEtCollection,
                           collectionsToKeep, postfix):
    
        module = patMETs.clone(
            metSource = cms.InputTag(pfMEtCollection),
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
  
