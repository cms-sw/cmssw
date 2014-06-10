import FWCore.ParameterSet.Config as cms

from FWCore.GuiBrowsers.ConfigToolBase import *

import PhysicsTools.PatAlgos.tools.helpers as configtools
from PhysicsTools.PatAlgos.tools.trigTools import _addEventContent

from PhysicsTools.PatUtils.patPFMETCorrections_cff import *
import RecoMET.METProducers.METSigParams_cfi as jetResolutions
from PhysicsTools.PatAlgos.producersLayer1.metProducer_cfi import patMETs

class RunMEtUncertainties(ConfigToolBase):

    """ Shift energy of electrons, photons, muons, tau-jets and other jets
    reconstructed in the event up/down,
    in order to estimate effect of energy scale uncertainties on MET
   """
    _label='runMEtUncertainties'
    _defaultParameters=dicttypes.SortedKeysDict()
    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters, 'electronCollection', cms.InputTag('selectedPatElectrons'),
	                  "Input electron collection", Type=cms.InputTag, acceptNoneValue=True)
	self.addParameter(self._defaultParameters, 'photonCollection', None, # CV: set to empty InputTag to avoid double-counting wrt. selectedPatElectrons collection
	                  "Input photon collection", Type=cms.InputTag, acceptNoneValue=True)
	self.addParameter(self._defaultParameters, 'muonCollection', cms.InputTag('selectedPatMuons'),
                          "Input muon collection", Type=cms.InputTag, acceptNoneValue=True)
	self.addParameter(self._defaultParameters, 'tauCollection', cms.InputTag('selectedPatTaus'),
                          "Input tau collection", Type=cms.InputTag, acceptNoneValue=True)
	self.addParameter(self._defaultParameters, 'jetCollection', cms.InputTag('selectedPatJets'),
                          "Input jet collection", Type=cms.InputTag)
	self.addParameter(self._defaultParameters, 'dRjetCleaning', 0.5,
                          "Eta-phi distance for extra jet cleaning", Type=float)
        self.addParameter(self._defaultParameters, 'jetCorrLabel', "L3Absolute",
                          "NOTE: use 'L3Absolute' for MC/'L2L3Residual' for Data", Type=str)
	self.addParameter(self._defaultParameters, 'doSmearJets', True,
                          "Flag to enable/disable jet smearing to better match MC to Data", Type=bool)
        self.addParameter(self._defaultParameters, 'makeType1corrPFMEt', True,
                          "Flag to enable/disable sequence for Type 1 corrected PFMEt", Type=bool)
        self.addParameter(self._defaultParameters, 'makeType1p2corrPFMEt', True,
                          "Flag to enable/disable sequence for Type 1 + 2 corrected PFMEt", Type=bool)
        self.addParameter(self._defaultParameters, 'makePFMEtByMVA', False,
                          "Flag to enable/disable sequence for MVA-based PFMEt", Type=bool)
        self.addParameter(self._defaultParameters, 'makeNoPileUpPFMEt', False,
                          "Flag to enable/disable sequence for no-PU PFMEt", Type=bool)
        self.addParameter(self._defaultParameters, 'doApplyType0corr', False,
                          "Flag to enable/disable usage of Type-0 MET corrections", Type=bool)
        self.addParameter(self._defaultParameters, 'sysShiftCorrParameter', None,
                          "MET sys. shift correction parameters", Type=cms.PSet)
        self.addParameter(self._defaultParameters, 'doApplySysShiftCorr', False,
                          "Flag to enable/disable usage of MET sys. shift corrections", Type=bool)
	self.addParameter(self._defaultParameters, 'jetSmearFileName', 'PhysicsTools/PatUtils/data/pfJetResolutionMCtoDataCorrLUT.root',
                          "Name of ROOT file containing histogram with jet smearing factors", Type=str)
        self.addParameter(self._defaultParameters, 'jetSmearHistogram', 'pfJetResolutionMCtoDataCorrLUT',
                          "Name of histogram with jet smearing factors", Type=str)
	self.addParameter(self._defaultParameters, 'pfCandCollection', cms.InputTag('particleFlow'),
                          "Input PFCandidate collection", Type=cms.InputTag)
	self.addParameter(self._defaultParameters, 'jetCorrPayloadName', 'AK5PF',
                          "Use AK5PF for PFJets, AK5Calo for CaloJets", Type=str)
	self.addParameter(self._defaultParameters, 'varyByNsigmas', 1.0,
                          "Number of standard deviations by which energies are varied", Type=float)
        self.addParameter(self._defaultParameters, 'addToPatDefaultSequence', True,
                          "Flag to enable/disable that metUncertaintySequence is inserted into patDefaultSequence", Type=bool)
        self.addParameter(self._defaultParameters, 'outputModule', 'out',
                          "Module label of PoolOutputModule (empty label indicates no PoolOutputModule is to be configured)", Type=str)
        self.addParameter(self._defaultParameters, 'postfix', '',
                          "Technical parameter to identify the resulting sequence and its modules (allows multiple calls in a job)", Type=str)
        self._parameters=copy.deepcopy(self._defaultParameters)
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

    def _addSmearedJets(self, process, jetCollection, smearedJetCollectionName_parts,
                        jetSmearFileName, jetSmearHistogram, varyByNsigmas,
                        shiftBy = None, postfix = ""):

        smearedJets = cms.EDProducer("SmearedPATJetProducer",
            src = cms.InputTag(jetCollection),
            dRmaxGenJetMatch = cms.string('TMath::Min(0.5, 0.1 + 0.3*TMath::Exp(-0.05*(genJetPt - 10.)))'),
            sigmaMaxGenJetMatch = cms.double(5.),
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
            skipCorrJetPtThreshold = cms.double(1.e-2)
        )
        if shiftBy is not None:
            setattr(smearedJets, "shiftBy", cms.double(shiftBy*varyByNsigmas))
        smearedJetCollection = \
          self._addModuleToSequence(process, smearedJets,
                                    smearedJetCollectionName_parts,
                                    getattr(process, "metUncertaintySequence"+postfix), postfix)

        return smearedJetCollection

    def _propagateMEtUncertainties(self, process,
                                   particleCollection, particleType, shiftType, particleCollectionShiftUp, particleCollectionShiftDown,
                                   metProducer, sequence, postfix):

        # produce MET correction objects
        # (sum of differences in four-momentum between original and up/down shifted particle collection)
        moduleMETcorrShiftUp = cms.EDProducer("ShiftedParticleMETcorrInputProducer",
            srcOriginal = cms.InputTag(particleCollection),
            srcShifted = cms.InputTag(particleCollectionShiftUp)
        )
        moduleMETcorrShiftUpName = "patPFMETcorr%s%sUp" % (particleType, shiftType)
        moduleMETcorrShiftUpName += postfix
        setattr(process, moduleMETcorrShiftUpName, moduleMETcorrShiftUp)
        sequence += moduleMETcorrShiftUp
        moduleMETcorrShiftDown = moduleMETcorrShiftUp.clone(
            srcShifted = cms.InputTag(particleCollectionShiftDown)
        )
        moduleMETcorrShiftDownName = "patPFMETcorr%s%sDown" % (particleType, shiftType)
        moduleMETcorrShiftDownName += postfix
        setattr(process, moduleMETcorrShiftDownName, moduleMETcorrShiftDown)
        sequence += moduleMETcorrShiftDown

        # propagate effects of up/down shifts to MET
        moduleMETshiftUp = metProducer.clone(
            src = cms.InputTag(metProducer.label()),
            srcType1Corrections = cms.VInputTag(
                cms.InputTag(moduleMETcorrShiftUpName)
            )
        )
        metProducerLabel = metProducer.label()
        if postfix != "":
            if metProducerLabel[-len(postfix):] == postfix:
                metProducerLabel = metProducerLabel[0:-len(postfix)]
            else:
                raise StandardError("Tried to remove postfix %s from label %s, but it wasn't there" % (postfix, metProducerLabel))
        moduleMETshiftUpName = "%s%s%sUp" % (metProducerLabel, particleType, shiftType)
        moduleMETshiftUpName += postfix
        setattr(process, moduleMETshiftUpName, moduleMETshiftUp)
        sequence += moduleMETshiftUp
        moduleMETshiftDown = moduleMETshiftUp.clone(
            srcType1Corrections = cms.VInputTag(
                cms.InputTag(moduleMETcorrShiftDownName)
            )
        )
        moduleMETshiftDownName = "%s%s%sDown" % (metProducerLabel, particleType, shiftType)
        moduleMETshiftDownName += postfix
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
                                       electronCollection,
                                       photonCollection,
                                       muonCollection,
                                       tauCollection,
                                       jetCollection, cleanedJetCollection, lastJetCollection,
                                       jetCollectionResUp, jetCollectionResDown,
                                       varyByNsigmas,
                                       postfix):

        shiftedParticlesForMEtUncertainties = cms.Sequence()
        setattr(process, "shiftedParticlesForMEtUncertainties"+postfix, shiftedParticlesForMEtUncertainties)

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
            jetCorrInputFileName = cms.FileInPath('PhysicsTools/PatUtils/data/Summer12_V2_DATA_AK5PF_UncertaintySources.txt'),
            jetCorrUncertaintyTag = cms.string("SubTotalDataMC"),
            addResidualJES = cms.bool(True),
            jetCorrLabelUpToL3 = cms.string("ak5PFL1FastL2L3"),
            jetCorrLabelUpToL3Res = cms.string("ak5PFL1FastL2L3Residual"),
            shiftBy = cms.double(+1.*varyByNsigmas)
        )
        jetCollectionEnUpForRawMEt = \
          self._addModuleToSequence(process, jetsEnUpForRawMEt,
                                    [ "shifted", jetCollection.value(), "EnUpForRawMEt" ],
                                    shiftedParticlesForMEtUncertainties, postfix)
        shiftedParticleCollections['jetCollectionEnUpForRawMEt'] = jetCollectionEnUpForRawMEt
        collectionsToKeep.append(jetCollectionEnUpForRawMEt)
        jetsEnDownForRawMEt = jetsEnUpForRawMEt.clone(
            shiftBy = cms.double(-1.*varyByNsigmas)
        )
        jetCollectionEnDownForRawMEt = \
          self._addModuleToSequence(process, jetsEnDownForRawMEt,
                                    [ "shifted", jetCollection.value(), "EnDownForRawMEt" ],
                                    shiftedParticlesForMEtUncertainties, postfix)
        shiftedParticleCollections['jetCollectionEnDownForRawMEt'] = jetCollectionEnDownForRawMEt
        collectionsToKeep.append(jetCollectionEnDownForRawMEt)

        jetsEnUpForCorrMEt = jetsEnUpForRawMEt.clone(
            addResidualJES = cms.bool(False)
        )
        jetCollectionEnUpForCorrMEt = \
          self._addModuleToSequence(process, jetsEnUpForCorrMEt,
                                    [ "shifted", jetCollection.value(), "EnUpForCorrMEt" ],
                                    shiftedParticlesForMEtUncertainties, postfix)
        shiftedParticleCollections['jetCollectionEnUpForCorrMEt'] = jetCollectionEnUpForCorrMEt
        collectionsToKeep.append(jetCollectionEnUpForCorrMEt)
        jetsEnDownForCorrMEt = jetsEnUpForCorrMEt.clone(
            shiftBy = cms.double(-1.*varyByNsigmas)
        )
        jetCollectionEnDownForCorrMEt = \
          self._addModuleToSequence(process, jetsEnDownForCorrMEt,
                                    [ "shifted", jetCollection.value(), "EnDownForCorrMEt" ],
                                    shiftedParticlesForMEtUncertainties, postfix)
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
                                        shiftedParticlesForMEtUncertainties, postfix)
            shiftedParticleCollections['electronCollectionEnUp'] = electronCollectionEnUp
            collectionsToKeep.append(electronCollectionEnUp)
            electronsEnDown = electronsEnUp.clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            )
            electronCollectionEnDown = \
              self._addModuleToSequence(process, electronsEnDown,
                                        [ "shifted", electronCollection.value(), "EnDown" ],
                                        shiftedParticlesForMEtUncertainties, postfix)
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
                        binSelection = cms.string('isEB = true'),
                        binUncertainty = cms.double(0.01)
                    ),
                    cms.PSet(
                        binSelection = cms.string('isEB = false'),
                        binUncertainty = cms.double(0.025)
                    ),
                ),
                shiftBy = cms.double(+1.*varyByNsigmas)
            )
            photonCollectionEnUp = \
              self._addModuleToSequence(process, photonsEnUp,
                                        [ "shifted", photonCollection.value(), "EnUp" ],
                                        shiftedParticlesForMEtUncertainties, postfix)
            shiftedParticleCollections['photonCollectionEnUp'] = photonCollectionEnUp
            collectionsToKeep.append(photonCollectionEnUp)
            photonsEnDown = photonsEnUp.clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            )
            photonCollectionEnDown = \
              self._addModuleToSequence(process, photonsEnDown,
                                        [ "shifted", photonCollection.value(), "EnDown" ],
                                        shiftedParticlesForMEtUncertainties, postfix)
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
                uncertainty = cms.double(0.002),
                shiftBy = cms.double(+1.*varyByNsigmas)
            )
            muonCollectionEnUp = \
              self._addModuleToSequence(process, muonsEnUp,
                                        [ "shifted", muonCollection.value(), "EnUp" ],
                                        shiftedParticlesForMEtUncertainties, postfix)
            shiftedParticleCollections['muonCollectionEnUp'] = muonCollectionEnUp
            collectionsToKeep.append(muonCollectionEnUp)
            muonsEnDown = muonsEnUp.clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            )
            muonCollectionEnDown = \
              self._addModuleToSequence(process, muonsEnDown,
                                        [ "shifted", muonCollection.value(), "EnDown" ],
                                        shiftedParticlesForMEtUncertainties, postfix)
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
                                        shiftedParticlesForMEtUncertainties, postfix)
            shiftedParticleCollections['tauCollectionEnUp'] = tauCollectionEnUp
            collectionsToKeep.append(tauCollectionEnUp)
            tausEnDown = tausEnUp.clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            )
            tauCollectionEnDown = \
              self._addModuleToSequence(process, tausEnDown,
                                        [ "shifted", tauCollection.value(), "EnDown" ],
                                        shiftedParticlesForMEtUncertainties, postfix)
            shiftedParticleCollections['tauCollectionEnDown'] = tauCollectionEnDown
            collectionsToKeep.append(tauCollectionEnDown)

        return ( shiftedParticleCollections, collectionsToKeep )

    def _addCorrPFMEt(self, process, metUncertaintySequence,
                      shiftedParticleCollections, pfCandCollection,
                      collectionsToKeep,
                      doSmearJets,
                      makeType1corrPFMEt,
                      makeType1p2corrPFMEt,
                      doApplyType0corr,
                      sysShiftCorrParameter,
                      doApplySysShiftCorr,
                      jetCorrLabel,
                      varyByNsigmas,
                      postfix):

        if not (makeType1corrPFMEt or makeType1p2corrPFMEt):
            return

        if not hasattr(process, 'producePatPFMETCorrections'):
            process.load("PhysicsTools.PatUtils.patPFMETCorrections_cff")

        # If with empty postfix, make a backup of
        # process.producePatPFMETCorrections, because the original
        # sequence will be modified later in this function
        if postfix == "":
            configtools.cloneProcessingSnippet(process, process.producePatPFMETCorrections, "OriginalReserved")
        else:
            if postfix == "OriginalReserved":
                raise ValueError("Postfix label '%s' is reserved for internal usage !!" % postfix)

            if hasattr(process, "producePatPFMETCorrectionsOriginalReserved"):
                configtools.cloneProcessingSnippet(process, process.producePatPFMETCorrectionsOriginalReserved, postfix, removePostfix="OriginalReserved")
            else:
                configtools.cloneProcessingSnippet(process, process.producePatPFMETCorrections, postfix)

        # add "nominal" (unshifted) pat::MET collections
        getattr(process, "pfCandsNotInJet"+postfix).bottomCollection = pfCandCollection
        getattr(process, "selectedPatJetsForMETtype1p2Corr"+postfix).src = shiftedParticleCollections['lastJetCollection']
        getattr(process, "selectedPatJetsForMETtype2Corr"+postfix).src = shiftedParticleCollections['lastJetCollection']


        if doApplySysShiftCorr:
            if not hasattr(process, 'pfMEtSysShiftCorrSequence'):
                process.load("JetMETCorrections.Type1MET.pfMETsysShiftCorrections_cfi")
            if postfix != "":
                configtools.cloneProcessingSnippet(process, process.pfMEtSysShiftCorrSequence, postfix)

            getattr(process, "pfMEtSysShiftCorr"+postfix).parameter = sysShiftCorrParameter
            metUncertaintySequence += getattr(process, "pfMEtSysShiftCorrSequence"+postfix)

        metUncertaintySequence += getattr(process, "producePatPFMETCorrections"+postfix)

        patType1correctionsCentralValue = [ cms.InputTag('patPFJetMETtype1p2Corr'+postfix, 'type1') ]
        if doApplyType0corr:
            patType1correctionsCentralValue.extend([ cms.InputTag('patPFMETtype0Corr'+postfix) ])
        if doApplySysShiftCorr:
            patType1correctionsCentralValue.extend([ cms.InputTag('pfMEtSysShiftCorr'+postfix) ])
        getattr(process, "patType1CorrectedPFMet"+postfix).srcType1Corrections = cms.VInputTag(patType1correctionsCentralValue)
        getattr(process, "patType1p2CorrectedPFMet"+postfix).srcType1Corrections = cms.VInputTag(patType1correctionsCentralValue)

        collectionsToKeep.extend([
            'patPFMet'+postfix,
            'patType1CorrectedPFMet'+postfix,
            'patType1p2CorrectedPFMet'+postfix])

        setattr(process, "selectedPatJetsForMETtype1p2CorrEnUp"+postfix,
          getattr(process, shiftedParticleCollections['jetCollectionEnUpForCorrMEt']).clone(
            src = cms.InputTag('selectedPatJetsForMETtype1p2Corr'+postfix)
        ))
        metUncertaintySequence += getattr(process, "selectedPatJetsForMETtype1p2CorrEnUp"+postfix)
        setattr(process, "selectedPatJetsForMETtype1p2CorrEnDown"+postfix,
          getattr(process, shiftedParticleCollections['jetCollectionEnDownForCorrMEt']).clone(
            src = cms.InputTag('selectedPatJetsForMETtype1p2Corr'+postfix)
        ))
        metUncertaintySequence += getattr(process, "selectedPatJetsForMETtype1p2CorrEnDown"+postfix)
        if makeType1p2corrPFMEt:
            setattr(process, "selectedPatJetsForMETtype2CorrEnUp"+postfix,
              getattr(process, shiftedParticleCollections['jetCollectionEnUpForCorrMEt']).clone(
                src = cms.InputTag('selectedPatJetsForMETtype2Corr'+postfix)
            ))
            metUncertaintySequence += getattr(process, "selectedPatJetsForMETtype2CorrEnUp"+postfix)
            setattr(process, "selectedPatJetsForMETtype2CorrEnDown"+postfix,
              getattr(process, shiftedParticleCollections['jetCollectionEnDownForCorrMEt']).clone(
                src = cms.InputTag('selectedPatJetsForMETtype2Corr'+postfix)
            ))
            metUncertaintySequence += getattr(process, "selectedPatJetsForMETtype2CorrEnDown"+postfix)

        if doSmearJets:
            setattr(process, "selectedPatJetsForMETtype1p2CorrResUp"+postfix,
              getattr(process, shiftedParticleCollections['jetCollectionResUp']).clone(
                src = cms.InputTag('selectedPatJetsForMETtype1p2Corr'+postfix)
            ))
            metUncertaintySequence += getattr(process, "selectedPatJetsForMETtype1p2CorrResUp"+postfix)
            setattr(process, "selectedPatJetsForMETtype1p2CorrResDown"+postfix,
              getattr(process, shiftedParticleCollections['jetCollectionResDown']).clone(
                src = cms.InputTag('selectedPatJetsForMETtype1p2Corr'+postfix)
            ))
            metUncertaintySequence += getattr(process, "selectedPatJetsForMETtype1p2CorrResDown"+postfix)
            if makeType1p2corrPFMEt:
                setattr(process, "selectedPatJetsForMETtype2CorrResUp"+postfix,
                  getattr(process, shiftedParticleCollections['jetCollectionResUp']).clone(
                    src = cms.InputTag('selectedPatJetsForMETtype2Corr'+postfix)
                ))
                metUncertaintySequence += getattr(process, "selectedPatJetsForMETtype2CorrResUp"+postfix)
                setattr(process, "selectedPatJetsForMETtype2CorrResDown"+postfix,
                  getattr(process, shiftedParticleCollections['jetCollectionResDown']).clone(
                    src = cms.InputTag('selectedPatJetsForMETtype2Corr'+postfix)
                ))
                metUncertaintySequence += getattr(process, "selectedPatJetsForMETtype2CorrResDown"+postfix)

        if doSmearJets:
            # apply MET smearing to "raw" (uncorrected) MET
            smearedPatPFMetSequence = cms.Sequence()
            setattr(process, "smearedPatPFMetSequence"+postfix, smearedPatPFMetSequence)
            setattr(process, "patPFMetForMEtUncertainty"+postfix, getattr(process, "patPFMet"+postfix).clone())
            smearedPatPFMetSequence += getattr(process, "patPFMetForMEtUncertainty"+postfix)
            setattr(process, "patPFMETcorrJetSmearing"+postfix, cms.EDProducer("ShiftedParticleMETcorrInputProducer",
                srcOriginal = cms.InputTag(shiftedParticleCollections['cleanedJetCollection']),
                srcShifted = cms.InputTag(shiftedParticleCollections['lastJetCollection'])
            ))
            smearedPatPFMetSequence += getattr(process, "patPFMETcorrJetSmearing"+postfix)
            getattr(process, "producePatPFMETCorrections"+postfix).replace(getattr(process, "patPFMet"+postfix), smearedPatPFMetSequence)
            setattr(process, "patPFMet"+postfix, getattr(process, "patType1CorrectedPFMet"+postfix).clone(
                src = cms.InputTag('patPFMetForMEtUncertainty'+postfix),
                srcType1Corrections = cms.VInputTag(
                    cms.InputTag('patPFMETcorrJetSmearing'+postfix)
                )
            ))
            smearedPatPFMetSequence += getattr(process, "patPFMet"+postfix)
            metUncertaintySequence += smearedPatPFMetSequence

        # propagate shifts in jet energy to "raw" (uncorrected) and Type 1 corrected MET
        metCollectionsUp_DownForRawMEt = \
            self._propagateMEtUncertainties(
              process, shiftedParticleCollections['lastJetCollection'], "Jet", "En",
              shiftedParticleCollections['jetCollectionEnUpForRawMEt'], shiftedParticleCollections['jetCollectionEnDownForRawMEt'],
              getattr(process, "patPFMet"+postfix), metUncertaintySequence, postfix)
        collectionsToKeep.extend(metCollectionsUp_DownForRawMEt)

        metCollectionsUp_DownForCorrMEt = \
            self._propagateMEtUncertainties(
              process, shiftedParticleCollections['lastJetCollection'], "Jet", "En",
              shiftedParticleCollections['jetCollectionEnUpForCorrMEt'], shiftedParticleCollections['jetCollectionEnDownForCorrMEt'],
              getattr(process, "patType1CorrectedPFMet"+postfix), metUncertaintySequence, postfix)
        collectionsToKeep.extend(metCollectionsUp_DownForCorrMEt)

        # propagate shifts in jet energy to Type 1 + 2 corrected MET
        if makeType1p2corrPFMEt:
            setattr(process, "patPFJetMETtype1p2CorrEnUp"+postfix, getattr(process, "patPFJetMETtype1p2Corr"+postfix).clone(
                src = cms.InputTag(getattr(process, "selectedPatJetsForMETtype1p2CorrEnUp"+postfix).label()),
                jetCorrLabel = cms.string(jetCorrLabel)
            ))
            metUncertaintySequence += getattr(process, "patPFJetMETtype1p2CorrEnUp"+postfix)
            setattr(process, "patPFJetMETtype1p2CorrEnDown"+postfix, getattr(process, "patPFJetMETtype1p2CorrEnUp"+postfix).clone(
                src = cms.InputTag(getattr(process, "selectedPatJetsForMETtype1p2CorrEnDown"+postfix).label())
            ))
            metUncertaintySequence += getattr(process, "patPFJetMETtype1p2CorrEnDown"+postfix)
            setattr(process, "patPFJetMETtype2CorrEnUp"+postfix, getattr(process, "patPFJetMETtype2Corr"+postfix).clone(
                src = cms.InputTag('selectedPatJetsForMETtype2CorrEnUp'+postfix)
            ))
            metUncertaintySequence += getattr(process, "patPFJetMETtype2CorrEnUp"+postfix)
            setattr(process, "patPFJetMETtype2CorrEnDown"+postfix,  getattr(process, "patPFJetMETtype2Corr"+postfix).clone(
                src = cms.InputTag('selectedPatJetsForMETtype2CorrEnDown'+postfix)
            ))
            metUncertaintySequence += getattr(process, "patPFJetMETtype2CorrEnDown"+postfix)

            patType1correctionsJetEnUp = [ cms.InputTag('patPFJetMETtype1p2CorrEnUp'+postfix, 'type1') ]
            if doApplyType0corr:
                patType1correctionsJetEnUp.extend([ cms.InputTag('patPFMETtype0Corr'+postfix) ])
            if doApplySysShiftCorr:
                patType1correctionsJetEnUp.extend([ cms.InputTag('pfMEtSysShiftCorr'+postfix) ])
            setattr(process, "patType1p2CorrectedPFMetJetEnUp"+postfix, getattr(process, "patType1p2CorrectedPFMet"+postfix).clone(
                srcType1Corrections = cms.VInputTag(patType1correctionsJetEnUp),
                srcUnclEnergySums = cms.VInputTag(
                    cms.InputTag('patPFJetMETtype1p2CorrEnUp'+postfix, 'type2' ),
                    cms.InputTag('patPFJetMETtype2CorrEnUp'+postfix,   'type2' ),
                    cms.InputTag('patPFJetMETtype1p2CorrEnUp'+postfix, 'offset'),
                    cms.InputTag('pfCandMETcorr'+postfix)
                )
            ))
            metUncertaintySequence += getattr(process, "patType1p2CorrectedPFMetJetEnUp"+postfix)
            collectionsToKeep.append('patType1p2CorrectedPFMetJetEnUp'+postfix)
            patType1correctionsJetEnDown = [ cms.InputTag('patPFJetMETtype1p2CorrEnDown'+postfix, 'type1') ]
            if doApplyType0corr:
                patType1correctionsJetEnDown.extend([ cms.InputTag('patPFMETtype0Corr'+postfix) ])
            if doApplySysShiftCorr:
                patType1correctionsJetEnDown.extend([ cms.InputTag('pfMEtSysShiftCorr'+postfix) ])
            setattr(process, "patType1p2CorrectedPFMetJetEnDown"+postfix, getattr(process, "patType1p2CorrectedPFMetJetEnUp"+postfix).clone(
                srcType1Corrections = cms.VInputTag(patType1correctionsJetEnDown),
                srcUnclEnergySums = cms.VInputTag(
                    cms.InputTag('patPFJetMETtype1p2CorrEnDown'+postfix, 'type2' ),
                    cms.InputTag('patPFJetMETtype2CorrEnDown'+postfix,   'type2' ),
                    cms.InputTag('patPFJetMETtype1p2CorrEnDown'+postfix, 'offset'),
                    cms.InputTag('pfCandMETcorr'+postfix)
                )
            ))
            metUncertaintySequence += getattr(process, "patType1p2CorrectedPFMetJetEnDown"+postfix)
            collectionsToKeep.append('patType1p2CorrectedPFMetJetEnDown'+postfix)

        if doSmearJets:
            # propagate shifts in jet resolution to "raw" (uncorrected) MET and Type 1 corrected MET
            for metProducer in [ getattr(process, "patPFMet"+postfix),
                                 getattr(process, "patType1CorrectedPFMet"+postfix) ]:

                metCollectionsUp_Down = \
                    self._propagateMEtUncertainties(
                      process, shiftedParticleCollections['lastJetCollection'], "Jet", "Res",
                      shiftedParticleCollections['jetCollectionResUp'], shiftedParticleCollections['jetCollectionResDown'],
                      metProducer, metUncertaintySequence, postfix)
                collectionsToKeep.extend(metCollectionsUp_Down)

            # propagate shifts in jet resolution to Type 1 + 2 corrected MET
            if makeType1p2corrPFMEt:
                setattr(process, "patPFJetMETtype1p2CorrResUp"+postfix, getattr(process, "patPFJetMETtype1p2Corr"+postfix).clone(
                    src = cms.InputTag(getattr(process, "selectedPatJetsForMETtype1p2CorrResUp"+postfix).label()),
                    jetCorrLabel = cms.string(jetCorrLabel)
                ))
                metUncertaintySequence += getattr(process, "patPFJetMETtype1p2CorrResUp"+postfix)
                setattr(process, "patPFJetMETtype1p2CorrResDown"+postfix, getattr(process, "patPFJetMETtype1p2CorrResUp"+postfix).clone(
                    src = cms.InputTag(getattr(process, "selectedPatJetsForMETtype1p2CorrResDown"+postfix).label())
                ))
                metUncertaintySequence += getattr(process, "patPFJetMETtype1p2CorrResDown"+postfix)
                setattr(process, "patPFJetMETtype2CorrResUp"+postfix, getattr(process, "patPFJetMETtype2Corr"+postfix).clone(
                    src = cms.InputTag('selectedPatJetsForMETtype2CorrResUp'+postfix)
                ))
                metUncertaintySequence += getattr(process, "patPFJetMETtype2CorrResUp"+postfix)
                setattr(process, "patPFJetMETtype2CorrResDown"+postfix, getattr(process, "patPFJetMETtype2Corr"+postfix).clone(
                    src = cms.InputTag('selectedPatJetsForMETtype2CorrResDown'+postfix)
                ))
                metUncertaintySequence += getattr(process, "patPFJetMETtype2CorrResDown"+postfix)

                patType1correctionsJetResUp = [ cms.InputTag('patPFJetMETtype1p2CorrResUp'+postfix, 'type1') ]
                if doApplyType0corr:
                    patType1correctionsJetResUp.extend([ cms.InputTag('patPFMETtype0Corr'+postfix) ])
                if doApplySysShiftCorr:
                    patType1correctionsJetResUp.extend([ cms.InputTag('pfMEtSysShiftCorr'+postfix) ])
                setattr(process, "patType1p2CorrectedPFMetJetResUp"+postfix,  getattr(process, "patType1p2CorrectedPFMet"+postfix).clone(
                    srcType1Corrections = cms.VInputTag(patType1correctionsJetResUp),
                    srcUnclEnergySums = cms.VInputTag(
                        cms.InputTag('patPFJetMETtype1p2CorrResUp'+postfix, 'type2' ),
                        cms.InputTag('patPFJetMETtype2CorrResUp'+postfix,   'type2' ),
                        cms.InputTag('patPFJetMETtype1p2CorrResUp'+postfix, 'offset'),
                        cms.InputTag('pfCandMETcorr'+postfix)
                    )
                ))
                metUncertaintySequence += getattr(process, "patType1p2CorrectedPFMetJetResUp"+postfix)
                collectionsToKeep.append('patType1p2CorrectedPFMetJetResUp'+postfix)
                patType1correctionsJetResDown = [ cms.InputTag('patPFJetMETtype1p2CorrResDown'+postfix, 'type1') ]
                if doApplyType0corr:
                    patType1correctionsJetResDown.extend([ cms.InputTag('patPFMETtype0Corr'+postfix) ])
                if doApplySysShiftCorr:
                    patType1correctionsJetResDown.extend([ cms.InputTag('pfMEtSysShiftCorr'+postfix) ])
                setattr(process, "patType1p2CorrectedPFMetJetResDown"+postfix, getattr(process, "patType1p2CorrectedPFMetJetResUp"+postfix).clone(
                    srcType1Corrections = cms.VInputTag(patType1correctionsJetResDown),
                    srcUnclEnergySums = cms.VInputTag(
                        cms.InputTag('patPFJetMETtype1p2CorrResDown'+postfix, 'type2' ),
                        cms.InputTag('patPFJetMETtype2CorrResDown'+postfix,   'type2' ),
                        cms.InputTag('patPFJetMETtype1p2CorrResDown'+postfix, 'offset'),
                        cms.InputTag('pfCandMETcorr'+postfix)
                    )
                ))
                metUncertaintySequence += getattr(process, "patType1p2CorrectedPFMetJetResDown"+postfix)
                collectionsToKeep.append('patType1p2CorrectedPFMetJetResDown'+postfix)

        #--------------------------------------------------------------------------------------------
        # shift "unclustered energy" (PFJets of Pt < 10 GeV plus PFCandidates not within jets)
        # and propagate effect of shift to (Type 1 as well as Type 1 + 2 corrected) MET
        #--------------------------------------------------------------------------------------------

        unclEnMETcorrections = [
            [ 'pfCandMETcorr'+postfix, [ '' ] ],
            [ 'patPFJetMETtype1p2Corr'+postfix, [ 'type2', 'offset' ] ],
            [ 'patPFJetMETtype2Corr'+postfix, [ 'type2' ] ],
        ]
        unclEnMETcorrectionsUp = []
        unclEnMETcorrectionsDown = []
        for srcUnclEnMETcorr in unclEnMETcorrections:
            moduleUnclEnMETcorrUp = cms.EDProducer("ShiftedMETcorrInputProducer",
                src = cms.VInputTag(
                    [ cms.InputTag(srcUnclEnMETcorr[0], instanceLabel) for instanceLabel in srcUnclEnMETcorr[1] ]
                ),
                uncertainty = cms.double(0.10),
                shiftBy = cms.double(+1.*varyByNsigmas)
            )
            baseName = srcUnclEnMETcorr[0]
            if postfix != "":
                if baseName[-len(postfix):] == postfix:
                    baseName = baseName[0:-len(postfix)]
                else:
                    raise StandardError("Tried to remove postfix %s from label %s, but it wasn't there" % (postfix, baseName))
            moduleUnclEnMETcorrUpName = "%sUnclusteredEnUp" % baseName
            moduleUnclEnMETcorrUpName += postfix
            setattr(process, moduleUnclEnMETcorrUpName, moduleUnclEnMETcorrUp)
            metUncertaintySequence += moduleUnclEnMETcorrUp
            unclEnMETcorrectionsUp.extend([ cms.InputTag(moduleUnclEnMETcorrUpName, instanceLabel)
                                            for instanceLabel in srcUnclEnMETcorr[1] ] )
            moduleUnclEnMETcorrDown = moduleUnclEnMETcorrUp.clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            )
            moduleUnclEnMETcorrDownName = "%sUnclusteredEnDown" % baseName
            moduleUnclEnMETcorrDownName += postfix
            setattr(process, moduleUnclEnMETcorrDownName, moduleUnclEnMETcorrDown)
            metUncertaintySequence += moduleUnclEnMETcorrDown
            unclEnMETcorrectionsDown.extend([ cms.InputTag(moduleUnclEnMETcorrDownName, instanceLabel)
                                              for instanceLabel in srcUnclEnMETcorr[1] ] )

        # propagate shifts in jet energy/resolution to "raw" (uncorrected) MET
        setattr(process, "patPFMetUnclusteredEnUp"+postfix, getattr(process, "patType1CorrectedPFMet"+postfix).clone(
            src = cms.InputTag('patPFMet'+postfix),
            srcType1Corrections = cms.VInputTag(unclEnMETcorrectionsUp)
        ))
        metUncertaintySequence += getattr(process, "patPFMetUnclusteredEnUp"+postfix)
        collectionsToKeep.append('patPFMetUnclusteredEnUp'+postfix)
        setattr(process, "patPFMetUnclusteredEnDown"+postfix, getattr(process, "patPFMetUnclusteredEnUp"+postfix).clone(
            srcType1Corrections = cms.VInputTag(unclEnMETcorrectionsDown)
        ))
        metUncertaintySequence += getattr(process, "patPFMetUnclusteredEnDown"+postfix)
        collectionsToKeep.append('patPFMetUnclusteredEnDown'+postfix)

        # propagate shifts in jet energy/resolution to Type 1 corrected MET
        setattr(process, "patType1CorrectedPFMetUnclusteredEnUp"+postfix, getattr(process, "patType1CorrectedPFMet"+postfix).clone(
            src = cms.InputTag('patType1CorrectedPFMet'+postfix),
            srcType1Corrections = cms.VInputTag(unclEnMETcorrectionsUp)
        ))
        metUncertaintySequence += getattr(process, "patType1CorrectedPFMetUnclusteredEnUp"+postfix)
        collectionsToKeep.append('patType1CorrectedPFMetUnclusteredEnUp'+postfix)
        setattr(process, "patType1CorrectedPFMetUnclusteredEnDown"+postfix, getattr(process, "patType1CorrectedPFMetUnclusteredEnUp"+postfix).clone(
            srcType1Corrections = cms.VInputTag(unclEnMETcorrectionsDown)
        ))
        metUncertaintySequence += getattr(process, "patType1CorrectedPFMetUnclusteredEnDown"+postfix)
        collectionsToKeep.append('patType1CorrectedPFMetUnclusteredEnDown'+postfix)

        # propagate shifts in jet energy/resolution to Type 1 + 2 corrected MET
        if makeType1p2corrPFMEt:
            setattr(process, "patType1p2CorrectedPFMetUnclusteredEnUp"+postfix, getattr(process, "patType1p2CorrectedPFMet"+postfix).clone(
                srcUnclEnergySums = cms.VInputTag(
                    cms.InputTag('patPFJetMETtype1p2Corr'+postfix,                'type2' ),
                    cms.InputTag('patPFJetMETtype1p2CorrUnclusteredEnUp'+postfix, 'type2' ),
                    cms.InputTag('patPFJetMETtype2Corr'+postfix,                  'type2' ),
                    cms.InputTag('patPFJetMETtype2CorrUnclusteredEnUp'+postfix,   'type2' ),
                    cms.InputTag('patPFJetMETtype1p2Corr'+postfix,                'offset'),
                    cms.InputTag('patPFJetMETtype1p2CorrUnclusteredEnUp'+postfix, 'offset'),
                    cms.InputTag('pfCandMETcorr'+postfix),
                    cms.InputTag('pfCandMETcorrUnclusteredEnUp'+postfix)
                )
            ))
            metUncertaintySequence += getattr(process, "patType1p2CorrectedPFMetUnclusteredEnUp"+postfix)
            collectionsToKeep.append('patType1p2CorrectedPFMetUnclusteredEnUp'+postfix)
            setattr(process, "patType1p2CorrectedPFMetUnclusteredEnDown"+postfix, getattr(process, "patType1p2CorrectedPFMetUnclusteredEnUp"+postfix).clone(
                srcUnclEnergySums = cms.VInputTag(
                    cms.InputTag('patPFJetMETtype1p2Corr'+postfix,                  'type2' ),
                    cms.InputTag('patPFJetMETtype1p2CorrUnclusteredEnDown'+postfix, 'type2' ),
                    cms.InputTag('patPFJetMETtype2Corr'+postfix,                    'type2' ),
                    cms.InputTag('patPFJetMETtype2CorrUnclusteredEnDown'+postfix,   'type2' ),
                    cms.InputTag('patPFJetMETtype1p2Corr'+postfix,                  'offset'),
                    cms.InputTag('patPFJetMETtype1p2CorrUnclusteredEnDown'+postfix, 'offset'),
                    cms.InputTag('pfCandMETcorr'+postfix),
                    cms.InputTag('pfCandMETcorrUnclusteredEnDown'+postfix)
                )
            ))
            metUncertaintySequence += getattr(process, "patType1p2CorrectedPFMetUnclusteredEnDown"+postfix)
            collectionsToKeep.append('patType1p2CorrectedPFMetUnclusteredEnDown'+postfix)

        #--------------------------------------------------------------------------------------------
        # propagate shifted electron/photon, muon and tau-jet energies to MET
        #--------------------------------------------------------------------------------------------

        metProducers = [ getattr(process, "patPFMet"+postfix),
                         getattr(process, "patType1CorrectedPFMet"+postfix) ]
        if makeType1p2corrPFMEt:
            metProducers.append( getattr(process, "patType1p2CorrectedPFMet"+postfix) )
        for metProducer in metProducers:

            if self._isValidInputTag(shiftedParticleCollections['electronCollection']):
                metCollectionsUp_Down = \
                    self._propagateMEtUncertainties(
                      process, shiftedParticleCollections['electronCollection'].value(), "Electron", "En",
                      shiftedParticleCollections['electronCollectionEnUp'], shiftedParticleCollections['electronCollectionEnDown'],
                      metProducer, metUncertaintySequence, postfix)
                collectionsToKeep.extend(metCollectionsUp_Down)

            if self._isValidInputTag(shiftedParticleCollections['photonCollection']):
                metCollectionsUp_Down = \
                    self._propagateMEtUncertainties(
                      process, shiftedParticleCollections['photonCollection'].value(), "Photon", "En",
                      shiftedParticleCollections['photonCollectionEnUp'], shiftedParticleCollections['photonCollectionEnDown'],
                      metProducer, metUncertaintySequence, postfix)
                collectionsToKeep.extend(metCollectionsUp_Down)

            if self._isValidInputTag(shiftedParticleCollections['muonCollection']):
                metCollectionsUp_Down = \
                    self._propagateMEtUncertainties(
                      process, shiftedParticleCollections['muonCollection'].value(), "Muon", "En",
                      shiftedParticleCollections['muonCollectionEnUp'], shiftedParticleCollections['muonCollectionEnDown'],
                      metProducer, metUncertaintySequence, postfix)
                collectionsToKeep.extend(metCollectionsUp_Down)

            if self._isValidInputTag(shiftedParticleCollections['tauCollection']):
                metCollectionsUp_Down = \
                    self._propagateMEtUncertainties(
                      process, shiftedParticleCollections['tauCollection'].value(), "Tau", "En",
                      shiftedParticleCollections['tauCollectionEnUp'], shiftedParticleCollections['tauCollectionEnDown'],
                      metProducer, metUncertaintySequence, postfix)
                collectionsToKeep.extend(metCollectionsUp_Down)

    def _addPFCandidatesForPFMEtInput(self, process, metUncertaintySequence,
                                      particleCollection, particleType, shiftType, particleCollectionShiftUp, particleCollectionShiftDown,
                                      dRmatch,
                                      pfCandCollection, postfix):

        srcUnshiftedObjects = particleCollection
        if isinstance(srcUnshiftedObjects, cms.InputTag):
            srcUnshiftedObjects = srcUnshiftedObjects.value()
        moduleShiftUp = cms.EDProducer("ShiftedPFCandidateProducerForPFMEtMVA",
            srcPFCandidates = pfCandCollection,
            srcUnshiftedObjects = cms.InputTag(srcUnshiftedObjects),
            srcShiftedObjects = cms.InputTag(particleCollectionShiftUp),
            dRmatch_PFCandidate = cms.double(dRmatch)
        )
        moduleNameShiftUp = "pfCandidates%s%sUpForMEtUncertainties" % (particleType, shiftType)
        moduleNameShiftUp += postfix
        setattr(process, moduleNameShiftUp, moduleShiftUp)
        metUncertaintySequence += moduleShiftUp

        moduleShiftDown = moduleShiftUp.clone(
            srcShiftedObjects = cms.InputTag(particleCollectionShiftDown)
        )
        moduleNameShiftDown = "pfCandidates%s%sDownForMEtUncertainties" % (particleType, shiftType)
        moduleNameShiftDown += postfix
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

    def _addPFMEtByMVA(self, process, metUncertaintySequence,
                       shiftedParticleCollections, pfCandCollection,
                       collectionsToKeep,
                       doSmearJets,
                       makePFMEtByMVA,
                       varyByNsigmas,
                       postfix):

        if not makePFMEtByMVA:
            return

        if not hasattr(process, "pfMEtMVA"):
            process.load("JetMETCorrections.METPUSubtraction.mvaPFMET_cff")

        lastUncorrectedJetCollectionForPFMEtByMVA = 'ak5PFJets'
        lastCorrectedJetCollectionForPFMEtByMVA = 'calibratedAK5PFJetsForPFMEtMVA'
        if postfix != "":
            configtools.cloneProcessingSnippet(process, process.pfMEtMVAsequence, postfix)
            lastCorrectedJetCollectionForPFMEtByMVA += postfix

        if doSmearJets:
            process.load("RecoJets.Configuration.GenJetParticles_cff")
            metUncertaintySequence += process.genParticlesForJetsNoNu
            process.load("RecoJets.Configuration.RecoGenJets_cff")
            metUncertaintySequence += process.ak5GenJetsNoNu
            setattr(process, "smearedUncorrectedJetsForPFMEtByMVA"+postfix, cms.EDProducer("SmearedPFJetProducer",
                src = cms.InputTag('ak5PFJets'),
                jetCorrLabel = cms.string("ak5PFL1FastL2L3"),
                dRmaxGenJetMatch = cms.string('TMath::Min(0.5, 0.1 + 0.3*TMath::Exp(-0.05*(genJetPt - 10.)))'),
                sigmaMaxGenJetMatch = cms.double(5.),
                inputFileName = cms.FileInPath('PhysicsTools/PatUtils/data/pfJetResolutionMCtoDataCorrLUT.root'),
                lutName = cms.string('pfJetResolutionMCtoDataCorrLUT'),
                jetResolutions = jetResolutions.METSignificance_params,
                skipRawJetPtThreshold = cms.double(10.), # GeV
                skipCorrJetPtThreshold = cms.double(1.e-2),
                srcGenJets = cms.InputTag('ak5GenJetsNoNu')
            ))
            metUncertaintySequence += getattr(process, "smearedUncorrectedJetsForPFMEtByMVA"+postfix)
            getattr(process, "calibratedAK5PFJetsForPFMEtMVA"+postfix).src = cms.InputTag('smearedUncorrectedJetsForPFMEtByMVA'+postfix)
            getattr(process, "pfMEtMVA"+postfix).srcUncorrJets = cms.InputTag('smearedUncorrectedJetsForPFMEtByMVA'+postfix)
            metUncertaintySequence += getattr(process, "calibratedAK5PFJetsForPFMEtMVA"+postfix)
            setattr(process, "smearedCorrectedJetsForPFMEtByMVA"+postfix, getattr(process, "smearedUncorrectedJetsForPFMEtByMVA"+postfix).clone(
                src = cms.InputTag('calibratedAK5PFJetsForPFMEtMVA'+postfix),
                jetCorrLabel = cms.string("")
            ))
            metUncertaintySequence += getattr(process, "smearedCorrectedJetsForPFMEtByMVA"+postfix)
            getattr(process, "pfMEtMVA"+postfix).srcCorrJets = cms.InputTag('smearedCorrectedJetsForPFMEtByMVA'+postfix)
            metUncertaintySequence += getattr(process, "pfMEtMVA"+postfix)
        else:
            metUncertaintySequence += getattr(process, "pfMEtMVAsequence"+postfix)
        self._addPATMEtProducer(process, metUncertaintySequence,
                                'pfMEtMVA'+postfix, 'patPFMetMVA', collectionsToKeep, postfix)

        for leptonCollection in [ [ 'Electron', 'En', 'electronCollection', 0.3 ],
                                  [ 'Photon',   'En', 'photonCollection',   0.3 ],
                                  [ 'Muon',     'En', 'muonCollection',     0.3 ],
                                  [ 'Tau',      'En', 'tauCollection',      0.3 ] ]:
            if self._isValidInputTag(shiftedParticleCollections[leptonCollection[2]]):
                pfCandCollectionLeptonShiftUp, pfCandCollectionLeptonShiftDown = \
                  self._addPFCandidatesForPFMEtInput(
                    process, metUncertaintySequence,
                    shiftedParticleCollections['%s' % leptonCollection[2]], leptonCollection[0], leptonCollection[1],
                    shiftedParticleCollections['%s%sUp' % (leptonCollection[2], leptonCollection[1])],
                    shiftedParticleCollections['%s%sDown' % (leptonCollection[2], leptonCollection[1])],
                    leptonCollection[3],
                    pfCandCollection, postfix)
                modulePFMEtLeptonShiftUp = getattr(process, "pfMEtMVA"+postfix).clone(
                    srcPFCandidates = cms.InputTag(pfCandCollectionLeptonShiftUp),
                    srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(
                      shiftedParticleCollections, leptonCollection[2], '%s%sUp' % (leptonCollection[2], leptonCollection[1]), postfix=postfix))
                )
                modulePFMEtLeptonShiftUpName = "pfMEtMVA%s%sUp" % (leptonCollection[0], leptonCollection[1])
                modulePFMEtLeptonShiftUpName += postfix
                setattr(process, modulePFMEtLeptonShiftUpName, modulePFMEtLeptonShiftUp)
                metUncertaintySequence += modulePFMEtLeptonShiftUp
                self._addPATMEtProducer(process, metUncertaintySequence,
                                        modulePFMEtLeptonShiftUpName, 'patPFMetMVA%s%sUp' % (leptonCollection[0], leptonCollection[1]), collectionsToKeep, postfix)
                modulePFMEtLeptonShiftDown = getattr(process, "pfMEtMVA"+postfix).clone(
                    srcPFCandidates = cms.InputTag(pfCandCollectionLeptonShiftDown),
                    srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(
                      shiftedParticleCollections, leptonCollection[2], '%s%sDown' % (leptonCollection[2], leptonCollection[1]), postfix=postfix))
                )
                modulePFMEtLeptonShiftDownName = "pfMEtMVA%s%sDown" % (leptonCollection[0], leptonCollection[1])
                modulePFMEtLeptonShiftDownName += postfix
                setattr(process, modulePFMEtLeptonShiftDownName, modulePFMEtLeptonShiftDown)
                metUncertaintySequence += modulePFMEtLeptonShiftDown
                self._addPATMEtProducer(process, metUncertaintySequence,
                                        modulePFMEtLeptonShiftDownName, 'patPFMetMVA%s%sDown' % (leptonCollection[0], leptonCollection[1]), collectionsToKeep, postfix)

        if self._isValidInputTag(shiftedParticleCollections['jetCollection']):
            setattr(process, "uncorrectedJetsEnUpForPFMEtByMVA"+postfix, cms.EDProducer("ShiftedPFJetProducer",
                src = cms.InputTag(lastUncorrectedJetCollectionForPFMEtByMVA),
                jetCorrInputFileName = cms.FileInPath('PhysicsTools/PatUtils/data/Summer12_V2_DATA_AK5PF_UncertaintySources.txt'),
                jetCorrUncertaintyTag = cms.string("SubTotalDataMC"),
                addResidualJES = cms.bool(True),
                jetCorrLabelUpToL3 = cms.string("ak5PFL1FastL2L3"),
                jetCorrLabelUpToL3Res = cms.string("ak5PFL1FastL2L3Residual"),
                shiftBy = cms.double(+1.*varyByNsigmas)
            ))
            metUncertaintySequence += getattr(process, "uncorrectedJetsEnUpForPFMEtByMVA"+postfix)
            setattr(process, "uncorrectedJetsEnDownForPFMEtByMVA"+postfix, getattr(process, "uncorrectedJetsEnUpForPFMEtByMVA"+postfix).clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            ))
            metUncertaintySequence += getattr(process, "uncorrectedJetsEnDownForPFMEtByMVA"+postfix)
            setattr(process, "correctedJetsEnUpForPFMEtByMVA"+postfix, getattr(process, "uncorrectedJetsEnUpForPFMEtByMVA"+postfix).clone(
                src = cms.InputTag(lastCorrectedJetCollectionForPFMEtByMVA),
                addResidualJES = cms.bool(False),
                shiftBy = cms.double(+1.*varyByNsigmas)
            ))
            metUncertaintySequence += getattr(process, "correctedJetsEnUpForPFMEtByMVA"+postfix)
            setattr(process, "correctedJetsEnDownForPFMEtByMVA"+postfix, getattr(process, "correctedJetsEnUpForPFMEtByMVA"+postfix).clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            ))
            metUncertaintySequence += getattr(process, "correctedJetsEnDownForPFMEtByMVA"+postfix)
            pfCandCollectionJetEnUp, pfCandCollectionJetEnDown = \
              self._addPFCandidatesForPFMEtInput(
                process, metUncertaintySequence,
                shiftedParticleCollections['lastJetCollection'], "Jet", "En",
                shiftedParticleCollections['jetCollectionEnUpForCorrMEt'], shiftedParticleCollections['jetCollectionEnDownForCorrMEt'],
                0.5,
                pfCandCollection, postfix)
            setattr(process, "pfMEtMVAJetEnUp"+postfix, getattr(process, "pfMEtMVA").clone(
                srcCorrJets = cms.InputTag('correctedJetsEnUpForPFMEtByMVA'+postfix),
                srcUncorrJets = cms.InputTag('uncorrectedJetsEnUpForPFMEtByMVA'+postfix),
                srcPFCandidates = cms.InputTag(pfCandCollectionJetEnUp),
                srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections, postfix=postfix))
            ))
            metUncertaintySequence += getattr(process, "pfMEtMVAJetEnUp"+postfix)
            self._addPATMEtProducer(process, metUncertaintySequence,
                                    'pfMEtMVAJetEnUp'+postfix, 'patPFMetMVAJetEnUp', collectionsToKeep, postfix)
            setattr(process, "pfMEtMVAJetEnDown"+postfix, getattr(process, "pfMEtMVA"+postfix).clone(
                srcCorrJets = cms.InputTag('correctedJetsEnDownForPFMEtByMVA'+postfix),
                srcUncorrJets = cms.InputTag('uncorrectedJetsEnDownForPFMEtByMVA'+postfix),
                srcPFCandidates = cms.InputTag(pfCandCollectionJetEnDown),
                srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections, postfix=postfix))
            ))
            metUncertaintySequence += getattr(process, "pfMEtMVAJetEnDown"+postfix)
            self._addPATMEtProducer(process, metUncertaintySequence,
                                    'pfMEtMVAJetEnDown'+postfix, 'patPFMetMVAJetEnDown', collectionsToKeep, postfix)

            if hasattr(process, "smearedUncorrectedJetsForPFMEtByMVA"+postfix):
                setattr(process, "uncorrectedJetsResUpForPFMEtByMVA"+postfix, getattr(process, "smearedUncorrectedJetsForPFMEtByMVA"+postfix).clone(
                    shiftBy = cms.double(-1.*varyByNsigmas)
                ))
                metUncertaintySequence += getattr(process, "uncorrectedJetsResUpForPFMEtByMVA"+postfix)
                setattr(process, "uncorrectedJetsResDownForPFMEtByMVA"+postfix, getattr(process, "smearedUncorrectedJetsForPFMEtByMVA"+postfix).clone(
                    shiftBy = cms.double(+1.*varyByNsigmas)
                ))
                metUncertaintySequence += getattr(process, "uncorrectedJetsResDownForPFMEtByMVA"+postfix)
                setattr(process, "correctedJetsResUpForPFMEtByMVA"+postfix, getattr(process, "smearedCorrectedJetsForPFMEtByMVA"+postfix).clone(
                    shiftBy = cms.double(-1.*varyByNsigmas)
                ))
                metUncertaintySequence += getattr(process, "correctedJetsResUpForPFMEtByMVA"+postfix)
                setattr(process, "correctedJetsResDownForPFMEtByMVA"+postfix, getattr(process, "smearedCorrectedJetsForPFMEtByMVA"+postfix).clone(
                    shiftBy = cms.double(+1.*varyByNsigmas)
                ))
                metUncertaintySequence += getattr(process, "correctedJetsResDownForPFMEtByMVA"+postfix)
                pfCandCollectionJetResUp, pfCandCollectionJetResDown = \
                  self._addPFCandidatesForPFMEtInput(
                    process, metUncertaintySequence,
                    shiftedParticleCollections['jetCollection'], "Jet", "Res",
                    shiftedParticleCollections['jetCollectionResUp'], shiftedParticleCollections['jetCollectionResDown'],
                    0.5,
                    pfCandCollection, postfix)
                setattr(process, "pfMEtMVAJetResUp"+postfix, getattr(process, "pfMEtMVA"+postfix).clone(
                    srcCorrJets = cms.InputTag('correctedJetsResUpForPFMEtByMVA'+postfix),
                    srcUncorrJets = cms.InputTag('uncorrectedJetsResUpForPFMEtByMVA'+postfix),
                    srcPFCandidates = cms.InputTag(pfCandCollectionJetResUp),
                    srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections, postfix=postfix))
                ))
                metUncertaintySequence += getattr(process, "pfMEtMVAJetResUp"+postfix)
                self._addPATMEtProducer(process, metUncertaintySequence,
                                       'pfMEtMVAJetResUp'+postfix, 'patPFMetMVAJetResUp', collectionsToKeep, postfix)
                setattr(process, "pfMEtMVAJetResDown"+postfix, getattr(process, "pfMEtMVA"+postfix).clone(
                    srcCorrJets = cms.InputTag('correctedJetsResDownForPFMEtByMVA'+postfix),
                    srcUncorrJets = cms.InputTag('uncorrectedJetsResDownForPFMEtByMVA'+postfix),
                    srcPFCandidates = cms.InputTag(pfCandCollectionJetResDown),
                    srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections, postfix=postfix))
                ))
                metUncertaintySequence += getattr(process, "pfMEtMVAJetResDown"+postfix)
                self._addPATMEtProducer(process, metUncertaintySequence,
                                        'pfMEtMVAJetResDown'+postfix, 'patPFMetMVAJetResDown', collectionsToKeep, postfix)

            setattr(process, "pfCandsNotInJetUnclusteredEnUpForPFMEtByMVA"+postfix, cms.EDProducer("ShiftedPFCandidateProducer",
                src = cms.InputTag('pfCandsNotInJet'),
                shiftBy = cms.double(+1.*varyByNsigmas),
                uncertainty = cms.double(0.10)
            ))
            metUncertaintySequence += getattr(process, "pfCandsNotInJetUnclusteredEnUpForPFMEtByMVA"+postfix)
            setattr(process, "pfCandsNotInJetUnclusteredEnDownForPFMEtByMVA"+postfix, getattr(process, "pfCandsNotInJetUnclusteredEnUpForPFMEtByMVA"+postfix).clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            ))
            metUncertaintySequence += getattr(process, "pfCandsNotInJetUnclusteredEnDownForPFMEtByMVA"+postfix)
            pfCandCollectionUnclusteredEnUp, pfCandCollectionUnclusteredEnDown = \
              self._addPFCandidatesForPFMEtInput(
                process, metUncertaintySequence,
                pfCandCollection, "Unclustered", "En",
                'pfCandsNotInJetUnclusteredEnUpForPFMEtByMVA'+postfix, 'pfCandsNotInJetUnclusteredEnDownForPFMEtByMVA'+postfix, #fixme MM
                0.01,
                pfCandCollection, postfix)
            setattr(process, "pfMEtMVAUnclusteredEnUp"+postfix, getattr(process, "pfMEtMVA"+postfix).clone(
                srcPFCandidates = cms.InputTag(pfCandCollectionUnclusteredEnUp),
                srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections, postfix=postfix))
            ))
            metUncertaintySequence += getattr(process, "pfMEtMVAUnclusteredEnUp"+postfix)
            self._addPATMEtProducer(process, metUncertaintySequence,
                                    'pfMEtMVAUnclusteredEnUp'+postfix, 'patPFMetMVAUnclusteredEnUp', collectionsToKeep, postfix)
            setattr(process, "pfMEtMVAUnclusteredEnDown"+postfix, getattr(process, "pfMEtMVA"+postfix).clone(
                srcPFCandidates = cms.InputTag(pfCandCollectionUnclusteredEnDown),
                srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections, postfix=postfix))
            ))
            metUncertaintySequence += getattr(process, "pfMEtMVAUnclusteredEnDown"+postfix)
            self._addPATMEtProducer(process, metUncertaintySequence,
                                    'pfMEtMVAUnclusteredEnDown'+postfix, 'patPFMetMVAUnclusteredEnDown', collectionsToKeep, postfix)

# IN HERE
    def _addNoPileUpPFMEt(self, process, metUncertaintySequence,
                        shiftedParticleCollections, pfCandCollection,
                        collectionsToKeep,
                        doSmearJets,
                        makeNoPileUpPFMEt,
                        varyByNsigmas,
                        postfix):

        if not makeNoPileUpPFMEt:
            return

        if not hasattr(process, "noPileUpPFMEt"):
            process.load("JetMETCorrections.METPUSubtraction.noPileUpPFMET_cff")

        lastUncorrectedJetCollectionForNoPileUpPFMEt = 'ak5PFJets'
        lastCorrectedJetCollectionForNoPileUpPFMEt = 'calibratedAK5PFJetsForNoPileUpPFMEt'
        if postfix != "":
            configtools.cloneProcessingSnippet(process, process.noPileUpPFMEtSequence, postfix)
            lastCorrectedJetCollectionForNoPileUpPFMEt+= postfix


        if doSmearJets:
            process.load("RecoJets.Configuration.GenJetParticles_cff")
            metUncertaintySequence += process.genParticlesForJetsNoNu
            process.load("RecoJets.Configuration.RecoGenJets_cff")
            metUncertaintySequence += process.ak5GenJetsNoNu
            setattr(process, "smearedUncorrectedJetsForNoPileUpPFMEt"+postfix, cms.EDProducer("SmearedPFJetProducer",
                src = cms.InputTag('ak5PFJets'),
                jetCorrLabel = cms.string("ak5PFL1FastL2L3"),
                dRmaxGenJetMatch = cms.string('TMath::Min(0.5, 0.1 + 0.3*TMath::Exp(-0.05*(genJetPt - 10.)))'),
                sigmaMaxGenJetMatch = cms.double(5.),
                inputFileName = cms.FileInPath('PhysicsTools/PatUtils/data/pfJetResolutionMCtoDataCorrLUT.root'),
                lutName = cms.string('pfJetResolutionMCtoDataCorrLUT'),
                jetResolutions = jetResolutions.METSignificance_params,
                skipRawJetPtThreshold = cms.double(10.), # GeV
                skipCorrJetPtThreshold = cms.double(1.e-2),
                srcGenJets = cms.InputTag('ak5GenJetsNoNu'),
                ##verbosity = cms.int32(1)
            ))
            metUncertaintySequence += getattr(process, "smearedUncorrectedJetsForNoPileUpPFMEt"+postfix)
            getattr(process, "calibratedAK5PFJetsForNoPileUpPFMEt"+postfix).src = cms.InputTag('smearedUncorrectedJetsForNoPileUpPFMEt'+postfix)
        metUncertaintySequence += getattr(process, "noPileUpPFMEtSequence"+postfix)
        self._addPATMEtProducer(process, metUncertaintySequence,
                                'noPileUpPFMEt'+postfix, 'patPFMetNoPileUp', collectionsToKeep, postfix)

        for leptonCollection in [ [ 'Electron', 'En', 'electronCollection', 0.3 ],
                                  [ 'Photon',   'En', 'photonCollection',   0.3 ],
                                  [ 'Muon',     'En', 'muonCollection',     0.3 ],
                                  [ 'Tau',      'En', 'tauCollection',      0.3 ] ]:
            if self._isValidInputTag(shiftedParticleCollections[leptonCollection[2]]):
                pfCandCollectionLeptonShiftUp, pfCandCollectionLeptonShiftDown = \
                  self._addPFCandidatesForPFMEtInput(
                    process, metUncertaintySequence,
                    shiftedParticleCollections['%s' % leptonCollection[2]], leptonCollection[0], leptonCollection[1],
                    shiftedParticleCollections['%s%sUp' % (leptonCollection[2], leptonCollection[1])], shiftedParticleCollections['%s%sDown' % (leptonCollection[2], leptonCollection[1])],
                    leptonCollection[3],
                    pfCandCollection, postfix)
                modulePFCandidateToVertexAssociationShiftUp = process.pfCandidateToVertexAssociation.clone(
                    PFCandidateCollection = cms.InputTag(pfCandCollectionLeptonShiftUp)
                )
                modulePFCandidateToVertexAssociationShiftUpName = "pfCandidateToVertexAssociation%s%sUp" % (leptonCollection[0], leptonCollection[1])
                modulePFCandidateToVertexAssociationShiftUpName += postfix
                setattr(process, modulePFCandidateToVertexAssociationShiftUpName, modulePFCandidateToVertexAssociationShiftUp)
                metUncertaintySequence += modulePFCandidateToVertexAssociationShiftUp
                modulePFMEtDataLeptonShiftUp = getattr(process, "noPileUpPFMEtData"+postfix).clone(
                    srcPFCandidates = cms.InputTag(pfCandCollectionLeptonShiftUp),
                    srcPFCandToVertexAssociations = cms.InputTag(modulePFCandidateToVertexAssociationShiftUpName)
                )
                modulePFMEtDataLeptonShiftUpName = "noPileUpPFMEtData%s%sUp" % (leptonCollection[0], leptonCollection[1])
                modulePFMEtDataLeptonShiftUpName += postfix
                setattr(process, modulePFMEtDataLeptonShiftUpName, modulePFMEtDataLeptonShiftUp)
                metUncertaintySequence += modulePFMEtDataLeptonShiftUp
                modulePFMEtLeptonShiftUp = getattr(process, "noPileUpPFMEt"+postfix).clone(
                    srcMVAMEtData = cms.InputTag(modulePFMEtDataLeptonShiftUpName),
                    srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(
                      shiftedParticleCollections, leptonCollection[2], '%s%sUp' % (leptonCollection[2], leptonCollection[1]), postfix=postfix))
                )
                modulePFMEtLeptonShiftUpName = "noPileUpPFMEt%s%sUp" % (leptonCollection[0], leptonCollection[1])
                modulePFMEtLeptonShiftUpName += postfix
                setattr(process, modulePFMEtLeptonShiftUpName, modulePFMEtLeptonShiftUp)
                metUncertaintySequence += modulePFMEtLeptonShiftUp
                self._addPATMEtProducer(process, metUncertaintySequence,
                                        modulePFMEtLeptonShiftUpName, 'patPFMetNoPileUp%s%sUp' % (leptonCollection[0], leptonCollection[1]), collectionsToKeep, postfix)
                modulePFCandidateToVertexAssociationShiftDown = modulePFCandidateToVertexAssociationShiftUp.clone(
                    PFCandidateCollection = cms.InputTag(pfCandCollectionLeptonShiftDown)
                )
                modulePFCandidateToVertexAssociationShiftDownName = "pfCandidateToVertexAssociation%s%sDown" % (leptonCollection[0], leptonCollection[1])
                modulePFCandidateToVertexAssociationShiftDownName += postfix
                setattr(process, modulePFCandidateToVertexAssociationShiftDownName, modulePFCandidateToVertexAssociationShiftDown)
                metUncertaintySequence += modulePFCandidateToVertexAssociationShiftDown
                modulePFMEtDataLeptonShiftDown = getattr(process, "noPileUpPFMEtData"+postfix).clone(
                    srcPFCandidates = cms.InputTag(pfCandCollectionLeptonShiftDown),
                    srcPFCandToVertexAssociations = cms.InputTag(modulePFCandidateToVertexAssociationShiftDownName)
                )
                modulePFMEtDataLeptonShiftDownName = "noPileUpPFMEtData%s%sDown" % (leptonCollection[0], leptonCollection[1])
                modulePFMEtDataLeptonShiftDownName += postfix
                setattr(process, modulePFMEtDataLeptonShiftDownName, modulePFMEtDataLeptonShiftDown)
                metUncertaintySequence += modulePFMEtDataLeptonShiftDown
                modulePFMEtLeptonShiftDown = getattr(process, "noPileUpPFMEt"+postfix).clone(
                    srcMVAMEtData = cms.InputTag(modulePFMEtDataLeptonShiftDownName),
                    srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(
                      shiftedParticleCollections, leptonCollection[2], '%s%sDown' % (leptonCollection[2], leptonCollection[1]), postfix=postfix))
                )
                modulePFMEtLeptonShiftDownName = "noPileUpPFMEt%s%sDown" % (leptonCollection[0], leptonCollection[1])
                modulePFMEtLeptonShiftDownName += postfix
                setattr(process, modulePFMEtLeptonShiftDownName, modulePFMEtLeptonShiftDown)
                metUncertaintySequence += modulePFMEtLeptonShiftDown
                self._addPATMEtProducer(process, metUncertaintySequence,
                                        modulePFMEtLeptonShiftDownName, 'patPFMetNoPileUp%s%sDown' % (leptonCollection[0], leptonCollection[1]), collectionsToKeep, postfix)

        if self._isValidInputTag(shiftedParticleCollections['jetCollection']):
            setattr(process, "uncorrectedJetsEnUpForNoPileUpPFMEt"+postfix, cms.EDProducer("ShiftedPFJetProducer",
                src = cms.InputTag(lastUncorrectedJetCollectionForNoPileUpPFMEt),
                jetCorrInputFileName = cms.FileInPath('PhysicsTools/PatUtils/data/Summer12_V2_DATA_AK5PF_UncertaintySources.txt'),
                jetCorrUncertaintyTag = cms.string("SubTotalDataMC"),
                addResidualJES = cms.bool(False),
                jetCorrLabelUpToL3 = cms.string("ak5PFL1FastL2L3"),
                jetCorrLabelUpToL3Res = cms.string("ak5PFL1FastL2L3Residual"),
                shiftBy = cms.double(+1.*varyByNsigmas),
                ##verbosity = cms.int32(1)
            ))
            metUncertaintySequence += getattr(process, "uncorrectedJetsEnUpForNoPileUpPFMEt"+postfix)
            setattr(process, "correctedJetsEnUpForNoPileUpPFMEt"+postfix, getattr(process, "uncorrectedJetsEnUpForNoPileUpPFMEt"+postfix).clone(
                src = cms.InputTag(lastCorrectedJetCollectionForNoPileUpPFMEt),
                addResidualJES = cms.bool(False),
                shiftBy = cms.double(+1.*varyByNsigmas)
            ))
            metUncertaintySequence += getattr(process, "correctedJetsEnUpForNoPileUpPFMEt"+postfix)
            setattr(process, "puJetIdForNoPileUpPFMEtJetEnUp"+postfix, getattr(process, "puJetIdForNoPileUpPFMEt"+postfix).clone(
                jets = cms.InputTag('correctedJetsEnUpForNoPileUpPFMEt'+postfix)
            ))
            metUncertaintySequence += getattr(process, "puJetIdForNoPileUpPFMEtJetEnUp"+postfix)
            setattr(process, "noPileUpPFMEtDataJetEnUp"+postfix, getattr(process, "noPileUpPFMEtData"+postfix).clone(
                srcJets = cms.InputTag('correctedJetsEnUpForNoPileUpPFMEt'+postfix),
                srcJetIds = cms.InputTag('puJetIdForNoPileUpPFMEtJetEnUp'+postfix, 'fullId')
            ))
            metUncertaintySequence += getattr(process, "noPileUpPFMEtDataJetEnUp"+postfix)
            setattr(process, "noPileUpPFMEtJetEnUp"+postfix, getattr(process, "noPileUpPFMEt"+postfix).clone(
                srcMVAMEtData = cms.InputTag('noPileUpPFMEtDataJetEnUp'+postfix),
                srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections, postfix=postfix))
            ))
            metUncertaintySequence += getattr(process, "noPileUpPFMEtJetEnUp"+postfix)
            self._addPATMEtProducer(process, metUncertaintySequence,
                                    'noPileUpPFMEtJetEnUp'+postfix, 'patPFMetNoPileUpJetEnUp', collectionsToKeep, postfix)
            setattr(process, "uncorrectedJetsEnDownForNoPileUpPFMEt"+postfix, getattr(process, "uncorrectedJetsEnUpForNoPileUpPFMEt"+postfix).clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            ))
            metUncertaintySequence += getattr(process, "uncorrectedJetsEnDownForNoPileUpPFMEt"+postfix)
            setattr(process, "correctedJetsEnDownForNoPileUpPFMEt"+postfix, getattr(process, "correctedJetsEnUpForNoPileUpPFMEt"+postfix).clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            ))
            metUncertaintySequence += getattr(process, "correctedJetsEnDownForNoPileUpPFMEt"+postfix)
            setattr(process, "puJetIdForNoPileUpPFMEtJetEnDown"+postfix, getattr(process, "puJetIdForNoPileUpPFMEt"+postfix).clone(
                jets = cms.InputTag('correctedJetsEnDownForNoPileUpPFMEt'+postfix)
            ))
            metUncertaintySequence += getattr(process, "puJetIdForNoPileUpPFMEtJetEnDown"+postfix)
            setattr(process, "noPileUpPFMEtDataJetEnDown"+postfix, getattr(process, "noPileUpPFMEtData"+postfix).clone(
                srcJets = cms.InputTag('correctedJetsEnDownForNoPileUpPFMEt'+postfix),
                srcJetIds = cms.InputTag('puJetIdForNoPileUpPFMEtJetEnDown'+postfix, 'fullId')
            ))
            metUncertaintySequence += getattr(process, "noPileUpPFMEtDataJetEnDown"+postfix)
            setattr(process, "noPileUpPFMEtJetEnDown"+postfix, getattr(process, "noPileUpPFMEt"+postfix).clone(
                srcMVAMEtData = cms.InputTag('noPileUpPFMEtDataJetEnDown'+postfix),
                srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections, postfix=postfix))
            ))
            metUncertaintySequence += getattr(process, "noPileUpPFMEtJetEnDown"+postfix)
            self._addPATMEtProducer(process, metUncertaintySequence,
                                    'noPileUpPFMEtJetEnDown'+postfix, 'patPFMetNoPileUpJetEnDown', collectionsToKeep, postfix)

            if hasattr(process, "smearedUncorrectedJetsForNoPileUpPFMEt"+postfix):
                setattr(process, "smearedCorrectedJetsForNoPileUpPFMEt"+postfix, getattr(process, "smearedUncorrectedJetsForNoPileUpPFMEt"+postfix).clone(
                    src = cms.InputTag('calibratedAK5PFJetsForNoPileUpPFMEt'+postfix),
                    jetCorrLabel = cms.string("")
                ))
                setattr(process, "correctedJetsResUpForNoPileUpPFMEt"+postfix, getattr(process, "smearedCorrectedJetsForNoPileUpPFMEt"+postfix).clone(
                    shiftBy = cms.double(-1.*varyByNsigmas)
                ))
                metUncertaintySequence += getattr(process, "correctedJetsResUpForNoPileUpPFMEt"+postfix)
                setattr(process, "correctedJetsResDownForNoPileUpPFMEt"+postfix, getattr(process, "smearedCorrectedJetsForNoPileUpPFMEt"+postfix).clone(
                    shiftBy = cms.double(+1.*varyByNsigmas)
                ))
                metUncertaintySequence += getattr(process, "correctedJetsResDownForNoPileUpPFMEt"+postfix)
                setattr(process, "puJetIdForNoPileUpPFMEtJetResUp"+postfix, getattr(process, "puJetIdForNoPileUpPFMEt"+postfix).clone(
                    jets = cms.InputTag('correctedJetsResUpForNoPileUpPFMEt'+postfix)
                ))
                metUncertaintySequence += getattr(process, "puJetIdForNoPileUpPFMEtJetResUp"+postfix)
                setattr(process, "noPileUpPFMEtDataJetResUp"+postfix, getattr(process, "noPileUpPFMEtData"+postfix).clone(
                    srcJets = cms.InputTag('correctedJetsResUpForNoPileUpPFMEt'+postfix),
                    srcJetIds = cms.InputTag('puJetIdForNoPileUpPFMEtJetResUp'+postfix, 'fullId')
                ))
                metUncertaintySequence += getattr(process, "noPileUpPFMEtDataJetResUp"+postfix)
                setattr(process, "noPileUpPFMEtJetResUp"+postfix, getattr(process, "noPileUpPFMEt"+postfix).clone(
                    srcMVAMEtData = cms.InputTag('noPileUpPFMEtDataJetResUp'+postfix),
                    srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections, postfix=postfix))
                ))
                metUncertaintySequence += getattr(process, "noPileUpPFMEtJetResUp"+postfix)
                self._addPATMEtProducer(process, metUncertaintySequence,
                                        'noPileUpPFMEtJetResUp'+postfix, 'patPFMetNoPileUpJetResUp', collectionsToKeep, postfix)
                setattr(process, "puJetIdForNoPileUpPFMEtJetResDown"+postfix, getattr(process, "puJetIdForNoPileUpPFMEt"+postfix).clone(
                    jets = cms.InputTag('correctedJetsResDownForNoPileUpPFMEt'+postfix)
                ))
                metUncertaintySequence += getattr(process, "puJetIdForNoPileUpPFMEtJetResDown"+postfix)
                setattr(process, "noPileUpPFMEtDataJetResDown"+postfix, getattr(process, "noPileUpPFMEtData"+postfix).clone(
                    srcJets = cms.InputTag('correctedJetsResDownForNoPileUpPFMEt'+postfix),
                    srcJetIds = cms.InputTag('puJetIdForNoPileUpPFMEtJetResDown'+postfix, 'fullId')
                ))
                metUncertaintySequence += getattr(process, "noPileUpPFMEtDataJetResDown"+postfix)
                setattr(process, "noPileUpPFMEtJetResDown"+postfix, getattr(process, "noPileUpPFMEt"+postfix).clone(
                    srcMVAMEtData = cms.InputTag('noPileUpPFMEtDataJetResDown'+postfix),
                    srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections, postfix=postfix))
                ))
                metUncertaintySequence += getattr(process, "noPileUpPFMEtJetResDown"+postfix)
                self._addPATMEtProducer(process, metUncertaintySequence,
                                        'noPileUpPFMEtJetResDown'+postfix, 'patPFMetNoPileUpJetResDown', collectionsToKeep, postfix)

            setattr(process, "pfCandsUnclusteredEnUpForNoPileUpPFMEt"+postfix, cms.EDProducer("ShiftedPFCandidateProducerForNoPileUpPFMEt",
                srcPFCandidates = cms.InputTag('particleFlow'),
                srcJets = cms.InputTag('calibratedAK5PFJetsForNoPileUpPFMEt'+postfix),
                jetCorrInputFileName = cms.FileInPath('PhysicsTools/PatUtils/data/Summer12_V2_DATA_AK5PF_UncertaintySources.txt'),
                jetCorrUncertaintyTag = cms.string("SubTotalDataMC"),
                minJetPt = cms.double(10.0),
                shiftBy = cms.double(+1.*varyByNsigmas),
                unclEnUncertainty = cms.double(0.10)
            ))
            metUncertaintySequence += getattr(process, "pfCandsUnclusteredEnUpForNoPileUpPFMEt"+postfix)
            setattr(process, "pfCandidateToVertexAssociationUnclusteredEnUpForNoPileUpPFMEt"+postfix, process.pfCandidateToVertexAssociation.clone(
                PFCandidateCollection = cms.InputTag('pfCandsUnclusteredEnUpForNoPileUpPFMEt'+postfix)
            ))
            metUncertaintySequence += getattr(process, "pfCandidateToVertexAssociationUnclusteredEnUpForNoPileUpPFMEt"+postfix)
            setattr(process, "noPileUpPFMEtDataUnclusteredEnUp"+postfix, getattr(process, "noPileUpPFMEtData"+postfix).clone(
                srcPFCandidates = cms.InputTag('pfCandsUnclusteredEnUpForNoPileUpPFMEt'+postfix),
                srcPFCandToVertexAssociations = cms.InputTag('pfCandidateToVertexAssociationUnclusteredEnUpForNoPileUpPFMEt'+postfix),
            ))
            metUncertaintySequence += getattr(process, "noPileUpPFMEtDataUnclusteredEnUp"+postfix)
            setattr(process, "noPileUpPFMEtUnclusteredEnUp"+postfix, getattr(process, "noPileUpPFMEt"+postfix).clone(
                srcMVAMEtData = cms.InputTag('noPileUpPFMEtDataUnclusteredEnUp'+postfix),
                srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections, postfix=postfix))
            ))
            metUncertaintySequence += getattr(process, "noPileUpPFMEtUnclusteredEnUp"+postfix)
            self._addPATMEtProducer(process, metUncertaintySequence,
                                    'noPileUpPFMEtUnclusteredEnUp'+postfix, 'patPFMetNoPileUpUnclusteredEnUp', collectionsToKeep, postfix)
            setattr(process, "pfCandsUnclusteredEnDownForNoPileUpPFMEt"+postfix, getattr(process, "pfCandsUnclusteredEnUpForNoPileUpPFMEt"+postfix).clone(
                shiftBy = cms.double(-1.*varyByNsigmas),
            ))
            metUncertaintySequence += getattr(process, "pfCandsUnclusteredEnDownForNoPileUpPFMEt"+postfix)
            setattr(process, "pfCandidateToVertexAssociationUnclusteredEnDownForNoPileUpPFMEt"+postfix, process.pfCandidateToVertexAssociation.clone(
                PFCandidateCollection = cms.InputTag('pfCandsUnclusteredEnDownForNoPileUpPFMEt'+postfix)
            ))
            metUncertaintySequence += getattr(process, "pfCandidateToVertexAssociationUnclusteredEnDownForNoPileUpPFMEt"+postfix)
            setattr(process, "noPileUpPFMEtDataUnclusteredEnDown"+postfix, getattr(process, "noPileUpPFMEtData"+postfix).clone(
                srcPFCandidates = cms.InputTag('pfCandsUnclusteredEnDownForNoPileUpPFMEt'+postfix),
                srcPFCandToVertexAssociations = cms.InputTag('pfCandidateToVertexAssociationUnclusteredEnDownForNoPileUpPFMEt'+postfix),
            ))
            metUncertaintySequence += getattr(process, "noPileUpPFMEtDataUnclusteredEnDown"+postfix)
            setattr(process, "noPileUpPFMEtUnclusteredEnDown"+postfix, getattr(process, "noPileUpPFMEt"+postfix).clone(
                srcMVAMEtData = cms.InputTag('noPileUpPFMEtDataUnclusteredEnDown'+postfix),
                srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections, postfix=postfix))
            ))
            metUncertaintySequence += getattr(process, "noPileUpPFMEtUnclusteredEnDown"+postfix)
            self._addPATMEtProducer(process, metUncertaintySequence,
                                    'noPileUpPFMEtUnclusteredEnDown'+postfix, 'patPFMetNoPileUpUnclusteredEnDown', collectionsToKeep, postfix)

    def __call__(self, process,
                 electronCollection      = None,
                 photonCollection        = None,
                 muonCollection          = None,
                 tauCollection           = None,
                 jetCollection           = None,
                 dRjetCleaning           = None,
                 jetCorrLabel            = None,
                 doSmearJets             = None,
                 makeType1corrPFMEt      = None,
                 makeType1p2corrPFMEt    = None,
                 makePFMEtByMVA          = None,
                 makeNoPileUpPFMEt       = None,
                 doApplyType0corr        = None,
                 sysShiftCorrParameter   = None,
                 doApplySysShiftCorr     = None,
                 jetSmearFileName        = None,
                 jetSmearHistogram       = None,
                 pfCandCollection        = None,
                 jetCorrPayloadName      = None,
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
        if dRjetCleaning is None:
            dRjetCleaning = self._defaultParameters['dRjetCleaning'].value
        if doSmearJets is None:
            doSmearJets = self._defaultParameters['doSmearJets'].value
        if makeType1corrPFMEt is None:
            makeType1corrPFMEt = self._defaultParameters['makeType1corrPFMEt'].value
        if makeType1p2corrPFMEt is None:
            makeType1p2corrPFMEt = self._defaultParameters['makeType1p2corrPFMEt'].value
        if makePFMEtByMVA is None:
            makePFMEtByMVA = self._defaultParameters['makePFMEtByMVA'].value
        if makeNoPileUpPFMEt is None:
            makeNoPileUpPFMEt = self._defaultParameters['makeNoPileUpPFMEt'].value
        if doApplyType0corr is None:
            doApplyType0corr = self._defaultParameters['doApplyType0corr'].value
        if sysShiftCorrParameter is None:
            sysShiftCorrParameter = self._defaultParameters['sysShiftCorrParameter'].value
        if doApplySysShiftCorr is None:
            doApplySysShiftCorr = self._defaultParameters['doApplySysShiftCorr'].value
        if sysShiftCorrParameter is None:
            if doApplySysShiftCorr:
                raise ValueError("MET sys. shift correction parameters must be specified explicitely !!")
            sysShiftCorrParameter = cms.PSet()
        if jetSmearFileName is None:
            jetSmearFileName = self._defaultParameters['jetSmearFileName'].value
        if jetSmearHistogram is None:
            jetSmearHistogram = self._defaultParameters['jetSmearHistogram'].value
        pfCandCollection = self._initializeInputTag(pfCandCollection, 'pfCandCollection')
        if jetCorrPayloadName is None:
            jetCorrPayloadName = self._defaultParameters['jetCorrPayloadName'].value
        if varyByNsigmas is None:
            varyByNsigmas = self._defaultParameters['varyByNsigmas'].value
        if  addToPatDefaultSequence is None:
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
        self.setParameter('dRjetCleaning', dRjetCleaning)
        self.setParameter('doSmearJets', doSmearJets)
        self.setParameter('makeType1corrPFMEt', makeType1corrPFMEt)
        self.setParameter('makeType1p2corrPFMEt', makeType1p2corrPFMEt)
        self.setParameter('makePFMEtByMVA', makePFMEtByMVA)
        self.setParameter('makeNoPileUpPFMEt', makeNoPileUpPFMEt)
        self.setParameter('doApplyType0corr', doApplyType0corr)
        self.setParameter('doApplySysShiftCorr', doApplySysShiftCorr)
        self.setParameter('sysShiftCorrParameter', sysShiftCorrParameter)
        self.setParameter('jetSmearFileName', jetSmearFileName)
        self.setParameter('jetSmearHistogram', jetSmearHistogram)
        self.setParameter('pfCandCollection', pfCandCollection)
        self.setParameter('jetCorrPayloadName', jetCorrPayloadName)
        self.setParameter('varyByNsigmas', varyByNsigmas)
        self.setParameter('addToPatDefaultSequence', addToPatDefaultSequence)
        self.setParameter('outputModule', outputModule)
        self.setParameter('postfix', postfix)

        self.apply(process)

    def toolCode(self, process):
        electronCollection = self._parameters['electronCollection'].value
        photonCollection = self._parameters['photonCollection'].value
        muonCollection = self._parameters['muonCollection'].value
        tauCollection = self._parameters['tauCollection'].value
        jetCollection = self._parameters['jetCollection'].value
        jetCorrLabel = self._parameters['jetCorrLabel'].value
        dRjetCleaning =  self._parameters['dRjetCleaning'].value
        doSmearJets = self._parameters['doSmearJets'].value
        makeType1corrPFMEt = self._parameters['makeType1corrPFMEt'].value
        makeType1p2corrPFMEt = self._parameters['makeType1p2corrPFMEt'].value
        makePFMEtByMVA = self._parameters['makePFMEtByMVA'].value
        makeNoPileUpPFMEt = self._parameters['makeNoPileUpPFMEt'].value
        doApplyType0corr = self._parameters['doApplyType0corr'].value
        sysShiftCorrParameter = self._parameters['sysShiftCorrParameter'].value
        doApplySysShiftCorr = self._parameters['doApplySysShiftCorr'].value
        jetSmearFileName = self._parameters['jetSmearFileName'].value
        jetSmearHistogram = self._parameters['jetSmearHistogram'].value
        pfCandCollection = self._parameters['pfCandCollection'].value
        jetCorrPayloadName = self._parameters['jetCorrPayloadName'].value
        varyByNsigmas = self._parameters['varyByNsigmas'].value
        addToPatDefaultSequence = self._parameters['addToPatDefaultSequence'].value
        outputModule = self._parameters['outputModule'].value
        postfix = self._parameters['postfix'].value

        metUncertaintySequence = cms.Sequence()
        setattr(process, "metUncertaintySequence"+postfix, metUncertaintySequence)

        collectionsToKeep = []

        # produce collection of jets not overlapping with reconstructed
        # electrons/photons, muons and tau-jet candidates
        jetsNotOverlappingWithLeptonsForMEtUncertainty = cms.EDProducer("PATJetCleaner",
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
                setattr(jetsNotOverlappingWithLeptonsForMEtUncertainty.checkOverlaps, collection[0], cms.PSet(
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
              self._addModuleToSequence(process, jetsNotOverlappingWithLeptonsForMEtUncertainty,
                                        [ jetCollection.value(), "NotOverlappingWithLeptonsForMEtUncertainty" ],
                                        metUncertaintySequence, postfix)
        cleanedJetCollection = lastJetCollection

        # smear jet energies to account for difference in jet resolutions between MC and Data
        # (cf. JME-10-014 PAS)
        jetCollectionResUp = None
        jetCollectionResDown = None
        if doSmearJets:
            lastJetCollection = \
              self._addSmearedJets(process, cleanedJetCollection, [ "smeared", jetCollection.value() ],
                                   jetSmearFileName, jetSmearHistogram, varyByNsigmas, postfix=postfix)

            jetCollectionResUp = \
              self._addSmearedJets(process, cleanedJetCollection, [ "smeared", jetCollection.value(), "ResUp" ],
                                   jetSmearFileName, jetSmearHistogram, varyByNsigmas,
                                   -1., postfix=postfix)
            collectionsToKeep.append(jetCollectionResUp)
            jetCollectionResDown = \
              self._addSmearedJets(process, cleanedJetCollection, [ "smeared", jetCollection.value(), "ResDown" ],
                                   jetSmearFileName, jetSmearHistogram, varyByNsigmas,
                                   +1., postfix=postfix)
            collectionsToKeep.append(jetCollectionResDown)

        collectionsToKeep.append(lastJetCollection)

        #--------------------------------------------------------------------------------------------
        # produce collection of electrons/photons, muons, tau-jet candidates and jets
        # shifted up/down in energy by their respective energy uncertainties
        #--------------------------------------------------------------------------------------------

        shiftedParticleCollections, addCollectionsToKeep = \
          self._addShiftedParticleCollections(process,
                                              electronCollection,
                                              photonCollection,
                                              muonCollection,
                                              tauCollection,
                                              jetCollection, cleanedJetCollection, lastJetCollection,
                                              jetCollectionResUp, jetCollectionResDown,
                                              varyByNsigmas,
                                              postfix)
        metUncertaintySequence += getattr(process, "shiftedParticlesForMEtUncertainties"+postfix)
        collectionsToKeep.extend(addCollectionsToKeep)

        #--------------------------------------------------------------------------------------------
        # propagate shifted particle energies to Type 1 and Type 1 + 2 corrected PFMET
        #--------------------------------------------------------------------------------------------

        self._addCorrPFMEt(process, metUncertaintySequence,
                           shiftedParticleCollections, pfCandCollection,
                           collectionsToKeep,
                           doSmearJets,
                           makeType1corrPFMEt,
                           makeType1p2corrPFMEt,
                           doApplyType0corr,
                           sysShiftCorrParameter,
                           doApplySysShiftCorr,
                           jetCorrLabel,
                           varyByNsigmas,
                           postfix)

        #--------------------------------------------------------------------------------------------
        # propagate shifted particle energies to MVA-based PFMET
        #--------------------------------------------------------------------------------------------

        self._addPFMEtByMVA(process, metUncertaintySequence,
                            shiftedParticleCollections, pfCandCollection,
                            collectionsToKeep,
                            doSmearJets,
                            makePFMEtByMVA,
                            varyByNsigmas,
                            postfix)

        #--------------------------------------------------------------------------------------------
        # propagate shifted particle energies to no-PU PFMET
        #--------------------------------------------------------------------------------------------

        self._addNoPileUpPFMEt(process, metUncertaintySequence,
                               shiftedParticleCollections, pfCandCollection,
                               collectionsToKeep,
                               doSmearJets,
                               makeNoPileUpPFMEt,
                               varyByNsigmas,
                               postfix)

        # insert metUncertaintySequence into patDefaultSequence
        if addToPatDefaultSequence and process.options.allowUnscheduled == False:
            if not hasattr(process, "patDefaultSequence"):
                raise ValueError("PAT default sequence is not defined !!")
            process.patDefaultSequence += metUncertaintySequence

        # add shifted + unshifted collections pf pat::Electrons/Photons,
        # Muons, Taus, Jets and MET to PAT-tuple event content
        if outputModule is not None and hasattr(process, outputModule):
            getattr(process, outputModule).outputCommands = _addEventContent(
                getattr(process, outputModule).outputCommands,
                [ 'keep *_%s_*_%s' % (collectionToKeep, process.name_()) for collectionToKeep in collectionsToKeep ])

runMEtUncertainties=RunMEtUncertainties()
