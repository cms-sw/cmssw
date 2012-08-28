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
        self._parameters=copy.deepcopy(self._defaultParameters)
        self._comment = ""
        
    def getDefaultParameters(self):
        return self._defaultParameters

    def _addModuleToSequence(self, process, module, moduleName_parts, sequence):

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

        setattr(process, moduleName, module)

        sequence += module
 
        return moduleName

    def _addSmearedJets(self, process, jetCollection, smearedJetCollectionName_parts,
                        jetSmearFileName, jetSmearHistogram, varyByNsigmas,
                        shiftBy = None):

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
                                    process.metUncertaintySequence)

        return smearedJetCollection
        
    def _propagateMEtUncertainties(self, process,
                                   particleCollection, particleType, shiftType, particleCollectionShiftUp, particleCollectionShiftDown,
                                   metProducer, sequence):

        # produce MET correction objects
        # (sum of differences in four-momentum between original and up/down shifted particle collection)
        moduleMETcorrShiftUp = cms.EDProducer("ShiftedParticleMETcorrInputProducer",
            srcOriginal = cms.InputTag(particleCollection),
            srcShifted = cms.InputTag(particleCollectionShiftUp)                                           
        )
        moduleMETcorrShiftUpName = "patPFMETcorr%s%sUp" % (particleType, shiftType)
        setattr(process, moduleMETcorrShiftUpName, moduleMETcorrShiftUp)
        sequence += moduleMETcorrShiftUp
        moduleMETcorrShiftDown = moduleMETcorrShiftUp.clone(
            srcShifted = cms.InputTag(particleCollectionShiftDown)                                           
        )
        moduleMETcorrShiftDownName = "patPFMETcorr%s%sDown" % (particleType, shiftType)
        setattr(process, moduleMETcorrShiftDownName, moduleMETcorrShiftDown)
        sequence += moduleMETcorrShiftDown

        # propagate effects of up/down shifts to MET
        moduleMETshiftUp = metProducer.clone(
            src = cms.InputTag(metProducer.label()),
            srcType1Corrections = cms.VInputTag(
                cms.InputTag(moduleMETcorrShiftUpName)
            )
        )
        moduleMETshiftUpName = "%s%s%sUp" % (metProducer.label(), particleType, shiftType)
        setattr(process, moduleMETshiftUpName, moduleMETshiftUp)
        sequence += moduleMETshiftUp
        moduleMETshiftDown = moduleMETshiftUp.clone(
            srcType1Corrections = cms.VInputTag(
                cms.InputTag(moduleMETcorrShiftDownName)
            )
        )
        moduleMETshiftDownName = "%s%s%sDown" % (metProducer.label(), particleType, shiftType)
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
                                       varyByNsigmas):

        process.shiftedParticlesForMEtUncertainties = cms.Sequence()
        
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
                                    process.shiftedParticlesForMEtUncertainties)
        shiftedParticleCollections['jetCollectionEnUpForRawMEt'] = jetCollectionEnUpForRawMEt
        collectionsToKeep.append(jetCollectionEnUpForRawMEt)
        jetsEnDownForRawMEt = jetsEnUpForRawMEt.clone(
            shiftBy = cms.double(-1.*varyByNsigmas)
        )
        jetCollectionEnDownForRawMEt = \
          self._addModuleToSequence(process, jetsEnDownForRawMEt,
                                    [ "shifted", jetCollection.value(), "EnDownForRawMEt" ],
                                    process.shiftedParticlesForMEtUncertainties)
        shiftedParticleCollections['jetCollectionEnDownForRawMEt'] = jetCollectionEnDownForRawMEt
        collectionsToKeep.append(jetCollectionEnDownForRawMEt)

        jetsEnUpForCorrMEt = jetsEnUpForRawMEt.clone(
            addResidualJES = cms.bool(False)
        )
        jetCollectionEnUpForCorrMEt = \
          self._addModuleToSequence(process, jetsEnUpForCorrMEt,
                                    [ "shifted", jetCollection.value(), "EnUpForCorrMEt" ],
                                    process.shiftedParticlesForMEtUncertainties)
        shiftedParticleCollections['jetCollectionEnUpForCorrMEt'] = jetCollectionEnUpForCorrMEt
        collectionsToKeep.append(jetCollectionEnUpForCorrMEt)
        jetsEnDownForCorrMEt = jetsEnUpForCorrMEt.clone(
            shiftBy = cms.double(-1.*varyByNsigmas)
        )
        jetCollectionEnDownForCorrMEt = \
          self._addModuleToSequence(process, jetsEnDownForCorrMEt,
                                    [ "shifted", jetCollection.value(), "EnDownForCorrMEt" ],
                                    process.shiftedParticlesForMEtUncertainties)
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
                                        process.shiftedParticlesForMEtUncertainties)
            shiftedParticleCollections['electronCollectionEnUp'] = electronCollectionEnUp
            collectionsToKeep.append(electronCollectionEnUp)
            electronsEnDown = electronsEnUp.clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            )
            electronCollectionEnDown = \
              self._addModuleToSequence(process, electronsEnDown,
                                        [ "shifted", electronCollection.value(), "EnDown" ],
                                        process.shiftedParticlesForMEtUncertainties)
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
                                        process.shiftedParticlesForMEtUncertainties)
            shiftedParticleCollections['photonCollectionEnUp'] = photonCollectionEnUp
            collectionsToKeep.append(photonCollectionEnUp)
            photonsEnDown = photonsEnUp.clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            )
            photonCollectionEnDown = \
              self._addModuleToSequence(process, photonsEnDown,
                                        [ "shifted", photonCollection.value(), "EnDown" ],
                                        process.shiftedParticlesForMEtUncertainties)
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
                                        process.shiftedParticlesForMEtUncertainties)
            shiftedParticleCollections['muonCollectionEnUp'] = muonCollectionEnUp
            collectionsToKeep.append(muonCollectionEnUp)
            muonsEnDown = muonsEnUp.clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            )
            muonCollectionEnDown = \
              self._addModuleToSequence(process, muonsEnDown,
                                        [ "shifted", muonCollection.value(), "EnDown" ],
                                        process.shiftedParticlesForMEtUncertainties)
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
                                        process.shiftedParticlesForMEtUncertainties)
            shiftedParticleCollections['tauCollectionEnUp'] = tauCollectionEnUp
            collectionsToKeep.append(tauCollectionEnUp)
            tausEnDown = tausEnUp.clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            )
            tauCollectionEnDown = \
              self._addModuleToSequence(process, tausEnDown,
                                        [ "shifted", tauCollection.value(), "EnDown" ],
                                        process.shiftedParticlesForMEtUncertainties)
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
                      varyByNsigmas):

        if not (makeType1corrPFMEt or makeType1p2corrPFMEt):
            return

        if not hasattr(process, 'producePatPFMETCorrections'):
            process.load("PhysicsTools.PatUtils.patPFMETCorrections_cff")
        
        # add "nominal" (unshifted) pat::MET collections        
        process.pfCandsNotInJet.bottomCollection = pfCandCollection        
        process.selectedPatJetsForMETtype1p2Corr.src = shiftedParticleCollections['lastJetCollection']
        process.selectedPatJetsForMETtype2Corr.src = shiftedParticleCollections['lastJetCollection']

        if doApplySysShiftCorr:
            if not hasattr(process, 'pfMEtSysShiftCorrSequence'):
                process.load("JetMETCorrections.Type1MET.pfMETsysShiftCorrections_cfi")
            process.pfMEtSysShiftCorr.parameter = sysShiftCorrParameter
            metUncertaintySequence += process.pfMEtSysShiftCorrSequence

        metUncertaintySequence += process.producePatPFMETCorrections
        
        patType1correctionsCentralValue = [ cms.InputTag('patPFJetMETtype1p2Corr', 'type1') ]
        if doApplyType0corr:
            patType1correctionsCentralValue.extend([ cms.InputTag('patPFMETtype0Corr') ])
        if doApplySysShiftCorr:
            patType1correctionsCentralValue.extend([ cms.InputTag('pfMEtSysShiftCorr') ])
        process.patType1CorrectedPFMet.srcType1Corrections = cms.VInputTag(patType1correctionsCentralValue)
        process.patType1p2CorrectedPFMet.srcType1Corrections = cms.VInputTag(patType1correctionsCentralValue)
        
        collectionsToKeep.extend([
            'patPFMet',
            'patType1CorrectedPFMet',
            'patType1p2CorrectedPFMet'])

        process.selectedPatJetsForMETtype1p2CorrEnUp = \
          getattr(process, shiftedParticleCollections['jetCollectionEnUpForCorrMEt']).clone(
            src = cms.InputTag('selectedPatJetsForMETtype1p2Corr')
        )
        metUncertaintySequence += process.selectedPatJetsForMETtype1p2CorrEnUp       
        process.selectedPatJetsForMETtype1p2CorrEnDown = \
          getattr(process, shiftedParticleCollections['jetCollectionEnDownForCorrMEt']).clone(
            src = cms.InputTag('selectedPatJetsForMETtype1p2Corr')
        )
        metUncertaintySequence += process.selectedPatJetsForMETtype1p2CorrEnDown
        if makeType1p2corrPFMEt:
            process.selectedPatJetsForMETtype2CorrEnUp = \
              getattr(process, shiftedParticleCollections['jetCollectionEnUpForCorrMEt']).clone(
                src = cms.InputTag('selectedPatJetsForMETtype2Corr')
            )
            metUncertaintySequence += process.selectedPatJetsForMETtype2CorrEnUp
            process.selectedPatJetsForMETtype2CorrEnDown = \
              getattr(process, shiftedParticleCollections['jetCollectionEnDownForCorrMEt']).clone(
                src = cms.InputTag('selectedPatJetsForMETtype2Corr')
            )
            metUncertaintySequence += process.selectedPatJetsForMETtype2CorrEnDown    

        if doSmearJets:
            process.selectedPatJetsForMETtype1p2CorrResUp = \
              getattr(process, shiftedParticleCollections['jetCollectionResUp']).clone(
                src = cms.InputTag('selectedPatJetsForMETtype1p2Corr')
            )
            metUncertaintySequence += process.selectedPatJetsForMETtype1p2CorrResUp
            process.selectedPatJetsForMETtype1p2CorrResDown = \
              getattr(process, shiftedParticleCollections['jetCollectionResDown']).clone(
                src = cms.InputTag('selectedPatJetsForMETtype1p2Corr')
            )
            metUncertaintySequence += process.selectedPatJetsForMETtype1p2CorrResDown
            if makeType1p2corrPFMEt:            
                process.selectedPatJetsForMETtype2CorrResUp = \
                  getattr(process, shiftedParticleCollections['jetCollectionResUp']).clone(
                    src = cms.InputTag('selectedPatJetsForMETtype2Corr')
                )
                metUncertaintySequence += process.selectedPatJetsForMETtype2CorrResUp            
                process.selectedPatJetsForMETtype2CorrResDown = \
                  getattr(process, shiftedParticleCollections['jetCollectionResDown']).clone(
                    src = cms.InputTag('selectedPatJetsForMETtype2Corr')
                )
                metUncertaintySequence += process.selectedPatJetsForMETtype2CorrResDown

        if doSmearJets:
            # apply MET smearing to "raw" (uncorrected) MET
            process.smearedPatPFMetSequence = cms.Sequence()
            process.patPFMetForMEtUncertainty = process.patPFMet.clone()
            process.smearedPatPFMetSequence += process.patPFMetForMEtUncertainty
            process.patPFMETcorrJetSmearing = cms.EDProducer("ShiftedParticleMETcorrInputProducer",
                srcOriginal = cms.InputTag(shiftedParticleCollections['cleanedJetCollection']),
                srcShifted = cms.InputTag(shiftedParticleCollections['lastJetCollection'])                                           
            )
            process.smearedPatPFMetSequence += process.patPFMETcorrJetSmearing
            process.producePatPFMETCorrections.replace(process.patPFMet, process.smearedPatPFMetSequence)
            process.patPFMet = process.patType1CorrectedPFMet.clone(
                src = cms.InputTag('patPFMetForMEtUncertainty'),
                srcType1Corrections = cms.VInputTag(
                    cms.InputTag('patPFMETcorrJetSmearing')
                )
            )
            process.smearedPatPFMetSequence += process.patPFMet
            metUncertaintySequence += process.smearedPatPFMetSequence 

        # propagate shifts in jet energy to "raw" (uncorrected) and Type 1 corrected MET
        metCollectionsUp_DownForRawMEt = \
            self._propagateMEtUncertainties(
              process, shiftedParticleCollections['lastJetCollection'], "Jet", "En",
              shiftedParticleCollections['jetCollectionEnUpForRawMEt'], shiftedParticleCollections['jetCollectionEnDownForRawMEt'],
              process.patPFMet, metUncertaintySequence)
        collectionsToKeep.extend(metCollectionsUp_DownForRawMEt)

        metCollectionsUp_DownForCorrMEt = \
            self._propagateMEtUncertainties(
              process, shiftedParticleCollections['lastJetCollection'], "Jet", "En",
              shiftedParticleCollections['jetCollectionEnUpForCorrMEt'], shiftedParticleCollections['jetCollectionEnDownForCorrMEt'],
              process.patType1CorrectedPFMet, metUncertaintySequence)
        collectionsToKeep.extend(metCollectionsUp_DownForCorrMEt)

        # propagate shifts in jet energy to Type 1 + 2 corrected MET
        if makeType1p2corrPFMEt:   
            process.patPFJetMETtype1p2CorrEnUp = process.patPFJetMETtype1p2Corr.clone(
                src = cms.InputTag(process.selectedPatJetsForMETtype1p2CorrEnUp.label()),
                jetCorrLabel = cms.string(jetCorrLabel)
            )
            metUncertaintySequence += process.patPFJetMETtype1p2CorrEnUp
            process.patPFJetMETtype1p2CorrEnDown = process.patPFJetMETtype1p2CorrEnUp.clone(
                src = cms.InputTag(process.selectedPatJetsForMETtype1p2CorrEnDown.label())
            )
            metUncertaintySequence += process.patPFJetMETtype1p2CorrEnDown
            process.patPFJetMETtype2CorrEnUp = process.patPFJetMETtype2Corr.clone(
                src = cms.InputTag('selectedPatJetsForMETtype2CorrEnUp')
            )
            process.metUncertaintySequence += process.patPFJetMETtype2CorrEnUp    
            process.patPFJetMETtype2CorrEnDown = process.patPFJetMETtype2Corr.clone(
                src = cms.InputTag('selectedPatJetsForMETtype2CorrEnDown')
            )
            process.metUncertaintySequence += process.patPFJetMETtype2CorrEnDown

            patType1correctionsJetEnUp = [ cms.InputTag('patPFJetMETtype1p2CorrEnUp', 'type1') ]        
            if doApplyType0corr:
                patType1correctionsJetEnUp.extend([ cms.InputTag('patPFMETtype0Corr') ])
            if doApplySysShiftCorr:
                patType1correctionsJetEnUp.extend([ cms.InputTag('pfMEtSysShiftCorr') ])            
            process.patType1p2CorrectedPFMetJetEnUp = process.patType1p2CorrectedPFMet.clone(
                srcType1Corrections = cms.VInputTag(patType1correctionsJetEnUp),
                srcUnclEnergySums = cms.VInputTag(
                    cms.InputTag('patPFJetMETtype1p2CorrEnUp', 'type2' ),
                    cms.InputTag('patPFJetMETtype2CorrEnUp',   'type2' ),
                    cms.InputTag('patPFJetMETtype1p2CorrEnUp', 'offset'),
                    cms.InputTag('pfCandMETcorr')                                    
                )
            )
            metUncertaintySequence += process.patType1p2CorrectedPFMetJetEnUp
            collectionsToKeep.append('patType1p2CorrectedPFMetJetEnUp')
            patType1correctionsJetEnDown = [ cms.InputTag('patPFJetMETtype1p2CorrEnDown', 'type1') ]
            if doApplyType0corr:
                patType1correctionsJetEnDown.extend([ cms.InputTag('patPFMETtype0Corr') ])
            if doApplySysShiftCorr:
                patType1correctionsJetEnDown.extend([ cms.InputTag('pfMEtSysShiftCorr') ])    
            process.patType1p2CorrectedPFMetJetEnDown = process.patType1p2CorrectedPFMetJetEnUp.clone(
                srcType1Corrections = cms.VInputTag(patType1correctionsJetEnDown),
                srcUnclEnergySums = cms.VInputTag(
                    cms.InputTag('patPFJetMETtype1p2CorrEnDown', 'type2' ),
                    cms.InputTag('patPFJetMETtype2CorrEnDown',   'type2' ),
                    cms.InputTag('patPFJetMETtype1p2CorrEnDown', 'offset'),
                    cms.InputTag('pfCandMETcorr')                                    
                )
            )
            metUncertaintySequence += process.patType1p2CorrectedPFMetJetEnDown
            collectionsToKeep.append('patType1p2CorrectedPFMetJetEnDown')

        if doSmearJets:
            # propagate shifts in jet resolution to "raw" (uncorrected) MET and Type 1 corrected MET
            for metProducer in [ process.patPFMet,
                                 process.patType1CorrectedPFMet ]:

                metCollectionsUp_Down = \
                    self._propagateMEtUncertainties(
                      process, shiftedParticleCollections['lastJetCollection'], "Jet", "Res",
                      shiftedParticleCollections['jetCollectionResUp'], shiftedParticleCollections['jetCollectionResDown'],
                      metProducer, process.metUncertaintySequence)
                collectionsToKeep.extend(metCollectionsUp_Down)
            
            # propagate shifts in jet resolution to Type 1 + 2 corrected MET
            if makeType1p2corrPFMEt:  
                process.patPFJetMETtype1p2CorrResUp = process.patPFJetMETtype1p2Corr.clone(
                    src = cms.InputTag(process.selectedPatJetsForMETtype1p2CorrResUp.label()),
                    jetCorrLabel = cms.string(jetCorrLabel)
                )
                metUncertaintySequence += process.patPFJetMETtype1p2CorrResUp
                process.patPFJetMETtype1p2CorrResDown = process.patPFJetMETtype1p2CorrResUp.clone(
                    src = cms.InputTag(process.selectedPatJetsForMETtype1p2CorrResDown.label())
                )
                metUncertaintySequence += process.patPFJetMETtype1p2CorrResDown
                process.patPFJetMETtype2CorrResUp = process.patPFJetMETtype2Corr.clone(
                    src = cms.InputTag('selectedPatJetsForMETtype2CorrResUp')
                )
                process.metUncertaintySequence += process.patPFJetMETtype2CorrResUp          
                process.patPFJetMETtype2CorrResDown = process.patPFJetMETtype2Corr.clone(
                    src = cms.InputTag('selectedPatJetsForMETtype2CorrResDown')
                )
                metUncertaintySequence += process.patPFJetMETtype2CorrResDown

                patType1correctionsJetResUp = [ cms.InputTag('patPFJetMETtype1p2CorrResUp', 'type1') ]
                if doApplyType0corr:
                    patType1correctionsJetResUp.extend([ cms.InputTag('patPFMETtype0Corr') ])
                if doApplySysShiftCorr:
                    patType1correctionsJetResUp.extend([ cms.InputTag('pfMEtSysShiftCorr') ])
                process.patType1p2CorrectedPFMetJetResUp = process.patType1p2CorrectedPFMet.clone(
                    srcType1Corrections = cms.VInputTag(patType1correctionsJetResUp),
                    srcUnclEnergySums = cms.VInputTag(
                        cms.InputTag('patPFJetMETtype1p2CorrResUp', 'type2' ),
                        cms.InputTag('patPFJetMETtype2CorrResUp',   'type2' ),
                        cms.InputTag('patPFJetMETtype1p2CorrResUp', 'offset'),
                        cms.InputTag('pfCandMETcorr')                                    
                    )
                )
                metUncertaintySequence += process.patType1p2CorrectedPFMetJetResUp
                collectionsToKeep.append('patType1p2CorrectedPFMetJetResUp')
                patType1correctionsJetResDown = [ cms.InputTag('patPFJetMETtype1p2CorrResDown', 'type1') ]
                if doApplyType0corr:
                    patType1correctionsJetResDown.extend([ cms.InputTag('patPFMETtype0Corr') ])
                if doApplySysShiftCorr:
                    patType1correctionsJetResDown.extend([ cms.InputTag('pfMEtSysShiftCorr') ])    
                process.patType1p2CorrectedPFMetJetResDown = process.patType1p2CorrectedPFMetJetResUp.clone(
                    srcType1Corrections = cms.VInputTag(patType1correctionsJetResDown),
                    srcUnclEnergySums = cms.VInputTag(
                        cms.InputTag('patPFJetMETtype1p2CorrResDown', 'type2' ),
                        cms.InputTag('patPFJetMETtype2CorrResDown',   'type2' ),
                        cms.InputTag('patPFJetMETtype1p2CorrResDown', 'offset'),
                        cms.InputTag('pfCandMETcorr')                                    
                    )
                )
                metUncertaintySequence += process.patType1p2CorrectedPFMetJetResDown
                collectionsToKeep.append('patType1p2CorrectedPFMetJetResDown')

        #--------------------------------------------------------------------------------------------
        # shift "unclustered energy" (PFJets of Pt < 10 GeV plus PFCandidates not within jets)
        # and propagate effect of shift to (Type 1 as well as Type 1 + 2 corrected) MET
        #--------------------------------------------------------------------------------------------

        unclEnMETcorrections = [
            [ 'pfCandMETcorr', [ '' ] ],
            [ 'patPFJetMETtype1p2Corr', [ 'type2', 'offset' ] ],
            [ 'patPFJetMETtype2Corr', [ 'type2' ] ],
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
            moduleUnclEnMETcorrUpName = "%sUnclusteredEnUp" % srcUnclEnMETcorr[0]
            setattr(process, moduleUnclEnMETcorrUpName, moduleUnclEnMETcorrUp)
            process.metUncertaintySequence += moduleUnclEnMETcorrUp
            unclEnMETcorrectionsUp.extend([ cms.InputTag(moduleUnclEnMETcorrUpName, instanceLabel)
                                            for instanceLabel in srcUnclEnMETcorr[1] ] )
            moduleUnclEnMETcorrDown = moduleUnclEnMETcorrUp.clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            )
            moduleUnclEnMETcorrDownName = "%sUnclusteredEnDown" % srcUnclEnMETcorr[0]
            setattr(process, moduleUnclEnMETcorrDownName, moduleUnclEnMETcorrDown)
            process.metUncertaintySequence += moduleUnclEnMETcorrDown
            unclEnMETcorrectionsDown.extend([ cms.InputTag(moduleUnclEnMETcorrDownName, instanceLabel)
                                              for instanceLabel in srcUnclEnMETcorr[1] ] )

        # propagate shifts in jet energy/resolution to "raw" (uncorrected) MET    
        process.patPFMetUnclusteredEnUp = process.patType1CorrectedPFMet.clone(
            src = cms.InputTag('patPFMet'),
            srcType1Corrections = cms.VInputTag(unclEnMETcorrectionsUp)
        )
        process.metUncertaintySequence += process.patPFMetUnclusteredEnUp
        collectionsToKeep.append('patPFMetUnclusteredEnUp')
        process.patPFMetUnclusteredEnDown = process.patPFMetUnclusteredEnUp.clone(
            srcType1Corrections = cms.VInputTag(unclEnMETcorrectionsDown)
        )
        process.metUncertaintySequence += process.patPFMetUnclusteredEnDown
        collectionsToKeep.append('patPFMetUnclusteredEnDown')        

        # propagate shifts in jet energy/resolution to Type 1 corrected MET
        process.patType1CorrectedPFMetUnclusteredEnUp = process.patType1CorrectedPFMet.clone(
            src = cms.InputTag('patType1CorrectedPFMet'),
            srcType1Corrections = cms.VInputTag(unclEnMETcorrectionsUp)
        )
        process.metUncertaintySequence += process.patType1CorrectedPFMetUnclusteredEnUp
        collectionsToKeep.append('patType1CorrectedPFMetUnclusteredEnUp')
        process.patType1CorrectedPFMetUnclusteredEnDown = process.patType1CorrectedPFMetUnclusteredEnUp.clone(
            srcType1Corrections = cms.VInputTag(unclEnMETcorrectionsDown)
        )
        process.metUncertaintySequence += process.patType1CorrectedPFMetUnclusteredEnDown
        collectionsToKeep.append('patType1CorrectedPFMetUnclusteredEnDown')
        
        # propagate shifts in jet energy/resolution to Type 1 + 2 corrected MET
        if makeType1p2corrPFMEt: 
            process.patType1p2CorrectedPFMetUnclusteredEnUp = process.patType1p2CorrectedPFMet.clone(
                srcUnclEnergySums = cms.VInputTag(
                    cms.InputTag('patPFJetMETtype1p2Corr',                'type2' ),
                    cms.InputTag('patPFJetMETtype1p2CorrUnclusteredEnUp', 'type2' ),
                    cms.InputTag('patPFJetMETtype2Corr',                  'type2' ),   
                    cms.InputTag('patPFJetMETtype2CorrUnclusteredEnUp',   'type2' ),
                    cms.InputTag('patPFJetMETtype1p2Corr',                'offset'),
                    cms.InputTag('patPFJetMETtype1p2CorrUnclusteredEnUp', 'offset'),
                    cms.InputTag('pfCandMETcorr'),
                    cms.InputTag('pfCandMETcorrUnclusteredEnUp')                                    
                )
            )
            metUncertaintySequence += process.patType1p2CorrectedPFMetUnclusteredEnUp
            collectionsToKeep.append('patType1p2CorrectedPFMetUnclusteredEnUp')
            process.patType1p2CorrectedPFMetUnclusteredEnDown = process.patType1p2CorrectedPFMetUnclusteredEnUp.clone(
                srcUnclEnergySums = cms.VInputTag(
                    cms.InputTag('patPFJetMETtype1p2Corr',                  'type2' ),
                    cms.InputTag('patPFJetMETtype1p2CorrUnclusteredEnDown', 'type2' ),
                    cms.InputTag('patPFJetMETtype2Corr',                    'type2' ),  
                    cms.InputTag('patPFJetMETtype2CorrUnclusteredEnDown',   'type2' ),
                    cms.InputTag('patPFJetMETtype1p2Corr',                  'offset'),
                    cms.InputTag('patPFJetMETtype1p2CorrUnclusteredEnDown', 'offset'),
                    cms.InputTag('pfCandMETcorr'),
                    cms.InputTag('pfCandMETcorrUnclusteredEnDown')                                    
                )
            )
            metUncertaintySequence += process.patType1p2CorrectedPFMetUnclusteredEnDown
            collectionsToKeep.append('patType1p2CorrectedPFMetUnclusteredEnDown')

        #--------------------------------------------------------------------------------------------    
        # propagate shifted electron/photon, muon and tau-jet energies to MET
        #--------------------------------------------------------------------------------------------

        metProducers = [ process.patPFMet,
                         process.patType1CorrectedPFMet ]
        if makeType1p2corrPFMEt:
            metProducers.append(process.patType1p2CorrectedPFMet)
        for metProducer in metProducers:
            
            if self._isValidInputTag(shiftedParticleCollections['electronCollection']):
                metCollectionsUp_Down = \
                    self._propagateMEtUncertainties(
                      process, shiftedParticleCollections['electronCollection'].value(), "Electron", "En",
                      shiftedParticleCollections['electronCollectionEnUp'], shiftedParticleCollections['electronCollectionEnDown'],
                      metProducer, metUncertaintySequence)
                collectionsToKeep.extend(metCollectionsUp_Down)

            if self._isValidInputTag(shiftedParticleCollections['photonCollection']):
                metCollectionsUp_Down = \
                    self._propagateMEtUncertainties(
                      process, shiftedParticleCollections['photonCollection'].value(), "Photon", "En",
                      shiftedParticleCollections['photonCollectionEnUp'], shiftedParticleCollections['photonCollectionEnDown'],
                      metProducer, metUncertaintySequence)
                collectionsToKeep.extend(metCollectionsUp_Down)
                
            if self._isValidInputTag(shiftedParticleCollections['muonCollection']):
                metCollectionsUp_Down = \
                    self._propagateMEtUncertainties(
                      process, shiftedParticleCollections['muonCollection'].value(), "Muon", "En",
                      shiftedParticleCollections['muonCollectionEnUp'], shiftedParticleCollections['muonCollectionEnDown'],
                      metProducer, metUncertaintySequence)
                collectionsToKeep.extend(metCollectionsUp_Down)

            if self._isValidInputTag(shiftedParticleCollections['tauCollection']):
                metCollectionsUp_Down = \
                    self._propagateMEtUncertainties(
                      process, shiftedParticleCollections['tauCollection'].value(), "Tau", "En",
                      shiftedParticleCollections['tauCollectionEnUp'], shiftedParticleCollections['tauCollectionEnDown'],
                      metProducer, metUncertaintySequence)
                collectionsToKeep.extend(metCollectionsUp_Down)

    def _addPFCandidatesForPFMEtInput(self, process, metUncertaintySequence,
                                      particleCollection, particleType, shiftType, particleCollectionShiftUp, particleCollectionShiftDown,
                                      dRmatch,
                                      pfCandCollection):

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
        setattr(process, moduleNameShiftUp, moduleShiftUp)
        metUncertaintySequence += moduleShiftUp

        moduleShiftDown = moduleShiftUp.clone(
            srcShiftedObjects = cms.InputTag(particleCollectionShiftDown)
        )
        moduleNameShiftDown = "pfCandidates%s%sDownForMEtUncertainties" % (particleType, shiftType)
        setattr(process, moduleNameShiftDown, moduleShiftDown)
        metUncertaintySequence += moduleShiftDown

        return ( moduleNameShiftUp, moduleNameShiftDown )

    def _getLeptonsForPFMEtInput(self, shiftedParticleCollections, substituteKeyUnshifted = None, substituteKeyShifted = None):
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
                           collectionsToKeep):
        
        module = patMETs.clone(
            metSource = cms.InputTag(pfMEtCollection),
            addMuonCorrections = cms.bool(False),
            genMETSource = cms.InputTag('genMetTrue')
        )
        setattr(process, patMEtCollection, module)
        metUncertaintySequence += module
        collectionsToKeep.append(patMEtCollection)
            
    def _addPFMEtByMVA(self, process, metUncertaintySequence,
                       shiftedParticleCollections, pfCandCollection,
                       collectionsToKeep,
                       doSmearJets,
                       makePFMEtByMVA,                       
                       varyByNsigmas):

        if not makePFMEtByMVA:
            return

        if not hasattr(process, "pfMEtMVA"):
            process.load("RecoMET.METProducers.mvaPFMET_cff")

        lastUncorrectedJetCollectionForPFMEtByMVA = 'ak5PFJets'
        lastCorrectedJetCollectionForPFMEtByMVA = 'calibratedAK5PFJetsForPFMEtMVA'
        
        if doSmearJets:
            process.load("RecoJets.Configuration.GenJetParticles_cff")
            metUncertaintySequence += process.genParticlesForJetsNoNu
            process.load("RecoJets.Configuration.RecoGenJets_cff")
            metUncertaintySequence += process.ak5GenJetsNoNu
            process.smearedUncorrectedJetsForPFMEtByMVA = cms.EDProducer("SmearedPFJetProducer",
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
            )
            metUncertaintySequence += process.smearedUncorrectedJetsForPFMEtByMVA
            process.calibratedAK5PFJetsForPFMEtMVA.src = cms.InputTag('smearedUncorrectedJetsForPFMEtByMVA')
            process.pfMEtMVA.srcUncorrJets = cms.InputTag('smearedUncorrectedJetsForPFMEtByMVA')
            metUncertaintySequence += process.calibratedAK5PFJetsForPFMEtMVA
            process.smearedCorrectedJetsForPFMEtByMVA = process.smearedUncorrectedJetsForPFMEtByMVA.clone(
                src = cms.InputTag('calibratedAK5PFJetsForPFMEtMVA'),
                jetCorrLabel = cms.string("")
            )
            metUncertaintySequence += process.smearedCorrectedJetsForPFMEtByMVA
            process.pfMEtMVA.srcCorrJets = cms.InputTag('smearedCorrectedJetsForPFMEtByMVA')
            metUncertaintySequence += process.pfMEtMVA            
        else:
            metUncertaintySequence += process.pfMEtMVAsequence
        self._addPATMEtProducer(process, metUncertaintySequence,
                                'pfMEtMVA', 'patPFMetMVA', collectionsToKeep)    

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
                    pfCandCollection)
                modulePFMEtLeptonShiftUp = process.pfMEtMVA.clone(
                    srcPFCandidates = cms.InputTag(pfCandCollectionLeptonShiftUp),
                    srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(
                      shiftedParticleCollections, leptonCollection[2], '%s%sUp' % (leptonCollection[2], leptonCollection[1])))
                )
                modulePFMEtLeptonShiftUpName = "pfMEtMVA%s%sUp" % (leptonCollection[0], leptonCollection[1])
                setattr(process, modulePFMEtLeptonShiftUpName, modulePFMEtLeptonShiftUp)
                metUncertaintySequence += modulePFMEtLeptonShiftUp
                self._addPATMEtProducer(process, metUncertaintySequence,
                                        modulePFMEtLeptonShiftUpName, 'patPFMetMVA%s%sUp' % (leptonCollection[0], leptonCollection[1]), collectionsToKeep)
                modulePFMEtLeptonShiftDown = process.pfMEtMVA.clone(
                    srcPFCandidates = cms.InputTag(pfCandCollectionLeptonShiftDown),
                    srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(
                      shiftedParticleCollections, leptonCollection[2], '%s%sDown' % (leptonCollection[2], leptonCollection[1])))
                )
                modulePFMEtLeptonShiftDownName = "pfMEtMVA%s%sDown" % (leptonCollection[0], leptonCollection[1])
                setattr(process, modulePFMEtLeptonShiftDownName, modulePFMEtLeptonShiftDown)
                metUncertaintySequence += modulePFMEtLeptonShiftDown
                self._addPATMEtProducer(process, metUncertaintySequence,
                                        modulePFMEtLeptonShiftDownName, 'patPFMetMVA%s%sDown' % (leptonCollection[0], leptonCollection[1]), collectionsToKeep)

        if self._isValidInputTag(shiftedParticleCollections['jetCollection']):            
            process.uncorrectedJetsEnUpForPFMEtByMVA = cms.EDProducer("ShiftedPFJetProducer",
                src = cms.InputTag(lastUncorrectedJetCollectionForPFMEtByMVA),
                jetCorrInputFileName = cms.FileInPath('PhysicsTools/PatUtils/data/Summer12_V2_DATA_AK5PF_UncertaintySources.txt'),
                jetCorrUncertaintyTag = cms.string("SubTotalDataMC"),
                addResidualJES = cms.bool(True),
                jetCorrLabelUpToL3 = cms.string("ak5PFL1FastL2L3"),
                jetCorrLabelUpToL3Res = cms.string("ak5PFL1FastL2L3Residual"),                               
                shiftBy = cms.double(+1.*varyByNsigmas)
            )
            metUncertaintySequence += process.uncorrectedJetsEnUpForPFMEtByMVA
            process.uncorrectedJetsEnDownForPFMEtByMVA = process.uncorrectedJetsEnUpForPFMEtByMVA.clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            )
            metUncertaintySequence += process.uncorrectedJetsEnDownForPFMEtByMVA
            process.correctedJetsEnUpForPFMEtByMVA = process.uncorrectedJetsEnUpForPFMEtByMVA.clone(
                src = cms.InputTag(lastCorrectedJetCollectionForPFMEtByMVA),
                addResidualJES = cms.bool(False),
                shiftBy = cms.double(+1.*varyByNsigmas)
            )
            metUncertaintySequence += process.correctedJetsEnUpForPFMEtByMVA
            process.correctedJetsEnDownForPFMEtByMVA = process.correctedJetsEnUpForPFMEtByMVA.clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            )
            metUncertaintySequence += process.correctedJetsEnDownForPFMEtByMVA
            pfCandCollectionJetEnUp, pfCandCollectionJetEnDown = \
              self._addPFCandidatesForPFMEtInput(
                process, metUncertaintySequence, 
                shiftedParticleCollections['lastJetCollection'], "Jet", "En",
                shiftedParticleCollections['jetCollectionEnUpForCorrMEt'], shiftedParticleCollections['jetCollectionEnDownForCorrMEt'],
                0.5,
                pfCandCollection)
            process.pfMEtMVAJetEnUp = process.pfMEtMVA.clone(
                srcCorrJets = cms.InputTag('correctedJetsEnUpForPFMEtByMVA'),
                srcUncorrJets = cms.InputTag('uncorrectedJetsEnUpForPFMEtByMVA'),
                srcPFCandidates = cms.InputTag(pfCandCollectionJetEnUp),
                srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections))
            )
            metUncertaintySequence += process.pfMEtMVAJetEnUp
            self._addPATMEtProducer(process, metUncertaintySequence,
                                    'pfMEtMVAJetEnUp', 'patPFMetMVAJetEnUp', collectionsToKeep)
            process.pfMEtMVAJetEnDown = process.pfMEtMVA.clone(
                srcCorrJets = cms.InputTag('correctedJetsEnDownForPFMEtByMVA'),
                srcUncorrJets = cms.InputTag('uncorrectedJetsEnDownForPFMEtByMVA'),
                srcPFCandidates = cms.InputTag(pfCandCollectionJetEnDown),
                srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections))
            )
            metUncertaintySequence += process.pfMEtMVAJetEnDown
            self._addPATMEtProducer(process, metUncertaintySequence,
                                    'pfMEtMVAJetEnDown', 'patPFMetMVAJetEnDown', collectionsToKeep)

            if hasattr(process, "smearedUncorrectedJetsForPFMEtByMVA"):
                process.uncorrectedJetsResUpForPFMEtByMVA = process.smearedUncorrectedJetsForPFMEtByMVA.clone(
                    shiftBy = cms.double(-1.*varyByNsigmas)
                )
                metUncertaintySequence += process.uncorrectedJetsResUpForPFMEtByMVA
                process.uncorrectedJetsResDownForPFMEtByMVA = process.smearedUncorrectedJetsForPFMEtByMVA.clone(
                    shiftBy = cms.double(+1.*varyByNsigmas)
                )
                metUncertaintySequence += process.uncorrectedJetsResDownForPFMEtByMVA
                process.correctedJetsResUpForPFMEtByMVA = process.smearedCorrectedJetsForPFMEtByMVA.clone(
                    shiftBy = cms.double(-1.*varyByNsigmas)
                )
                metUncertaintySequence += process.correctedJetsResUpForPFMEtByMVA
                process.correctedJetsResDownForPFMEtByMVA = process.smearedCorrectedJetsForPFMEtByMVA.clone(
                    shiftBy = cms.double(+1.*varyByNsigmas)
                )  
                metUncertaintySequence += process.correctedJetsResDownForPFMEtByMVA
                pfCandCollectionJetResUp, pfCandCollectionJetResDown = \
                  self._addPFCandidatesForPFMEtInput(
                    process, metUncertaintySequence,
                    shiftedParticleCollections['jetCollection'], "Jet", "Res",
                    shiftedParticleCollections['jetCollectionResUp'], shiftedParticleCollections['jetCollectionResDown'],
                    0.5,
                    pfCandCollection)
                process.pfMEtMVAJetResUp = process.pfMEtMVA.clone(
                    srcCorrJets = cms.InputTag('correctedJetsResUpForPFMEtByMVA'),
                    srcUncorrJets = cms.InputTag('uncorrectedJetsResUpForPFMEtByMVA'),
                    srcPFCandidates = cms.InputTag(pfCandCollectionJetResUp),
                    srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections))
                )
                metUncertaintySequence += process.pfMEtMVAJetResUp
                self._addPATMEtProducer(process, metUncertaintySequence,
                                       'pfMEtMVAJetResUp', 'patPFMetMVAJetResUp', collectionsToKeep)
                process.pfMEtMVAJetResDown = process.pfMEtMVA.clone(
                    srcCorrJets = cms.InputTag('correctedJetsResDownForPFMEtByMVA'),
                    srcUncorrJets = cms.InputTag('uncorrectedJetsResDownForPFMEtByMVA'),
                    srcPFCandidates = cms.InputTag(pfCandCollectionJetResDown),
                    srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections))
                )
                metUncertaintySequence += process.pfMEtMVAJetResDown
                self._addPATMEtProducer(process, metUncertaintySequence,
                                        'pfMEtMVAJetResDown', 'patPFMetMVAJetResDown', collectionsToKeep)
                        
            process.pfCandsNotInJetUnclusteredEnUpForPFMEtByMVA = cms.EDProducer("ShiftedPFCandidateProducer",
                src = cms.InputTag('pfCandsNotInJet'),
                shiftBy = cms.double(+1.*varyByNsigmas),
                uncertainty = cms.double(0.10)
            )
            metUncertaintySequence += process.pfCandsNotInJetUnclusteredEnUpForPFMEtByMVA
            process.pfCandsNotInJetUnclusteredEnDownForPFMEtByMVA = process.pfCandsNotInJetUnclusteredEnUpForPFMEtByMVA.clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            )
            metUncertaintySequence += process.pfCandsNotInJetUnclusteredEnDownForPFMEtByMVA
            pfCandCollectionUnclusteredEnUp, pfCandCollectionUnclusteredEnDown = \
              self._addPFCandidatesForPFMEtInput(
                process, metUncertaintySequence,
                pfCandCollection, "Unclustered", "En",
                'pfCandsNotInJetUnclusteredEnUpForPFMEtByMVA', 'pfCandsNotInJetUnclusteredEnDownForPFMEtByMVA',
                0.01,
                pfCandCollection)
            process.pfMEtMVAUnclusteredEnUp = process.pfMEtMVA.clone(
                srcPFCandidates = cms.InputTag(pfCandCollectionUnclusteredEnUp),
                srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections))
            )
            metUncertaintySequence += process.pfMEtMVAUnclusteredEnUp
            self._addPATMEtProducer(process, metUncertaintySequence,
                                    'pfMEtMVAUnclusteredEnUp', 'patPFMetMVAUnclusteredEnUp', collectionsToKeep)
            process.pfMEtMVAUnclusteredEnDown = process.pfMEtMVA.clone(
                srcPFCandidates = cms.InputTag(pfCandCollectionUnclusteredEnDown),
                srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections))
            )
            metUncertaintySequence += process.pfMEtMVAUnclusteredEnDown
            self._addPATMEtProducer(process, metUncertaintySequence,
                                    'pfMEtMVAUnclusteredEnDown', 'patPFMetMVAUnclusteredEnDown', collectionsToKeep)

    def _addNoPileUpPFMEt(self, process, metUncertaintySequence,
                        shiftedParticleCollections, pfCandCollection,
                        collectionsToKeep,
                        doSmearJets,
                        makeNoPileUpPFMEt,                       
                        varyByNsigmas):
        
        if not makeNoPileUpPFMEt:
            return

        if not hasattr(process, "noPileUpPFMEt"):
            process.load("JetMETCorrections.Type1MET.noPileUpPFMET_cff")

        lastUncorrectedJetCollectionForNoPileUpPFMEt = 'ak5PFJets'
        lastCorrectedJetCollectionForNoPileUpPFMEt = 'calibratedAK5PFJetsForNoPileUpMEt'
                
        if doSmearJets:
            process.load("RecoJets.Configuration.GenJetParticles_cff")
            metUncertaintySequence += process.genParticlesForJetsNoNu
            process.load("RecoJets.Configuration.RecoGenJets_cff")
            metUncertaintySequence += process.ak5GenJetsNoNu
            process.smearedUncorrectedJetsForNoPileUpPFMEt = cms.EDProducer("SmearedPFJetProducer",
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
            )
            metUncertaintySequence += process.smearedUncorrectedJetsForNoPileUpPFMEt
            process.calibratedAK5PFJetsForNoPileUpMEt.src = cms.InputTag('smearedUncorrectedJetsForNoPileUpPFMEt')
        metUncertaintySequence += process.noPileUpPFMEtSequence
        self._addPATMEtProducer(process, metUncertaintySequence,
                                'noPileUpPFMEt', 'patPFMetNoPileUp', collectionsToKeep)    

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
                    pfCandCollection)
                modulePFCandidateToVertexAssociationShiftUp = process.pfCandidateToVertexAssociation.clone(
                    PFCandidateCollection = cms.InputTag(pfCandCollectionLeptonShiftUp)
                )
                modulePFCandidateToVertexAssociationShiftUpName = "pfCandidateToVertexAssociation%s%sUp" % (leptonCollection[0], leptonCollection[1])
                setattr(process, modulePFCandidateToVertexAssociationShiftUpName, modulePFCandidateToVertexAssociationShiftUp)
                metUncertaintySequence += modulePFCandidateToVertexAssociationShiftUp
                modulePFMEtDataLeptonShiftUp = process.noPileUpPFMEtData.clone(
                    srcPFCandidates = cms.InputTag(pfCandCollectionLeptonShiftUp),
                    srcPFCandToVertexAssociations = cms.InputTag(modulePFCandidateToVertexAssociationShiftUpName)
                )
                modulePFMEtDataLeptonShiftUpName = "noPileUpPFMEtData%s%sUp" % (leptonCollection[0], leptonCollection[1])
                setattr(process, modulePFMEtDataLeptonShiftUpName, modulePFMEtDataLeptonShiftUp)
                metUncertaintySequence += modulePFMEtDataLeptonShiftUp
                modulePFMEtLeptonShiftUp = process.noPileUpPFMEt.clone(
                    srcMVAMEtData = cms.InputTag(modulePFMEtDataLeptonShiftUpName),
                    srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(
                      shiftedParticleCollections, leptonCollection[2], '%s%sUp' % (leptonCollection[2], leptonCollection[1])))
                )
                modulePFMEtLeptonShiftUpName = "noPileUpPFMEt%s%sUp" % (leptonCollection[0], leptonCollection[1])
                setattr(process, modulePFMEtLeptonShiftUpName, modulePFMEtLeptonShiftUp)
                metUncertaintySequence += modulePFMEtLeptonShiftUp
                self._addPATMEtProducer(process, metUncertaintySequence,
                                        modulePFMEtLeptonShiftUpName, 'patPFMetNoPileUp%s%sUp' % (leptonCollection[0], leptonCollection[1]), collectionsToKeep)
                modulePFCandidateToVertexAssociationShiftDown = modulePFCandidateToVertexAssociationShiftUp.clone(
                    PFCandidateCollection = cms.InputTag(pfCandCollectionLeptonShiftDown)
                )
                modulePFCandidateToVertexAssociationShiftDownName = "pfCandidateToVertexAssociation%s%sDown" % (leptonCollection[0], leptonCollection[1])
                setattr(process, modulePFCandidateToVertexAssociationShiftDownName, modulePFCandidateToVertexAssociationShiftDown)
                metUncertaintySequence += modulePFCandidateToVertexAssociationShiftDown                
                modulePFMEtDataLeptonShiftDown = process.noPileUpPFMEtData.clone(
                    srcPFCandidates = cms.InputTag(pfCandCollectionLeptonShiftDown),
                    srcPFCandToVertexAssociations = cms.InputTag(modulePFCandidateToVertexAssociationShiftDownName)
                )
                modulePFMEtDataLeptonShiftDownName = "noPileUpPFMEtData%s%sDown" % (leptonCollection[0], leptonCollection[1])
                setattr(process, modulePFMEtDataLeptonShiftDownName, modulePFMEtDataLeptonShiftDown)
                metUncertaintySequence += modulePFMEtDataLeptonShiftDown
                modulePFMEtLeptonShiftDown = process.noPileUpPFMEt.clone(
                    srcMVAMEtData = cms.InputTag(modulePFMEtDataLeptonShiftDownName),
                    srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(
                      shiftedParticleCollections, leptonCollection[2], '%s%sDown' % (leptonCollection[2], leptonCollection[1])))
                )
                modulePFMEtLeptonShiftDownName = "noPileUpPFMEt%s%sDown" % (leptonCollection[0], leptonCollection[1])
                setattr(process, modulePFMEtLeptonShiftDownName, modulePFMEtLeptonShiftDown)
                metUncertaintySequence += modulePFMEtLeptonShiftDown
                self._addPATMEtProducer(process, metUncertaintySequence,
                                        modulePFMEtLeptonShiftDownName, 'patPFMetNoPileUp%s%sDown' % (leptonCollection[0], leptonCollection[1]), collectionsToKeep)

        if self._isValidInputTag(shiftedParticleCollections['jetCollection']):            
            process.uncorrectedJetsEnUpForNoPileUpPFMEt = cms.EDProducer("ShiftedPFJetProducer",
                src = cms.InputTag(lastUncorrectedJetCollectionForNoPileUpPFMEt),
                jetCorrInputFileName = cms.FileInPath('PhysicsTools/PatUtils/data/Summer12_V2_DATA_AK5PF_UncertaintySources.txt'),
                jetCorrUncertaintyTag = cms.string("SubTotalDataMC"),
                addResidualJES = cms.bool(False),
                jetCorrLabelUpToL3 = cms.string("ak5PFL1FastL2L3"),
                jetCorrLabelUpToL3Res = cms.string("ak5PFL1FastL2L3Residual"),                               
                shiftBy = cms.double(+1.*varyByNsigmas),
                ##verbosity = cms.int32(1)
            )
            metUncertaintySequence += process.uncorrectedJetsEnUpForNoPileUpPFMEt           
            process.correctedJetsEnUpForNoPileUpPFMEt = process.uncorrectedJetsEnUpForNoPileUpPFMEt.clone(
                src = cms.InputTag(lastCorrectedJetCollectionForNoPileUpPFMEt),
                addResidualJES = cms.bool(False),
                shiftBy = cms.double(+1.*varyByNsigmas)
            )
            metUncertaintySequence += process.correctedJetsEnUpForNoPileUpPFMEt           
            process.puJetIdDataForNoPileUpMEtJetEnUp = process.puJetIdDataForNoPileUpMEt.clone(
                jets = cms.InputTag('correctedJetsEnUpForNoPileUpPFMEt')
            )
            metUncertaintySequence += process.puJetIdDataForNoPileUpMEtJetEnUp
            process.puJetIdForNoPileUpMEtJetEnUp = process.puJetIdForNoPileUpMEt.clone(
                jetids = cms.InputTag('puJetIdDataForNoPileUpMEtJetEnUp'),
                jets = cms.InputTag('correctedJetsEnUpForNoPileUpPFMEt')
            )
            metUncertaintySequence += process.puJetIdForNoPileUpMEtJetEnUp
            process.noPileUpPFMEtDataJetEnUp = process.noPileUpPFMEtData.clone(
                srcJets = cms.InputTag('correctedJetsEnUpForNoPileUpPFMEt'),
                srcJetIds = cms.InputTag('puJetIdForNoPileUpMEtJetEnUp', 'fullId')
            )
            metUncertaintySequence += process.noPileUpPFMEtDataJetEnUp
            process.noPileUpPFMEtJetEnUp = process.noPileUpPFMEt.clone(
                srcMVAMEtData = cms.InputTag('noPileUpPFMEtDataJetEnUp'),                                           
                srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections))
            )
            metUncertaintySequence += process.noPileUpPFMEtJetEnUp
            self._addPATMEtProducer(process, metUncertaintySequence,
                                    'noPileUpPFMEtJetEnUp', 'patPFMetNoPileUpJetEnUp', collectionsToKeep)
            process.uncorrectedJetsEnDownForNoPileUpPFMEt = process.uncorrectedJetsEnUpForNoPileUpPFMEt.clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            )
            metUncertaintySequence += process.uncorrectedJetsEnDownForNoPileUpPFMEt
            process.correctedJetsEnDownForNoPileUpPFMEt = process.correctedJetsEnUpForNoPileUpPFMEt.clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            )
            metUncertaintySequence += process.correctedJetsEnDownForNoPileUpPFMEt            
            process.puJetIdDataForNoPileUpMEtJetEnDown = process.puJetIdDataForNoPileUpMEt.clone(
                jets = cms.InputTag('correctedJetsEnDownForNoPileUpPFMEt')
            )
            metUncertaintySequence += process.puJetIdDataForNoPileUpMEtJetEnDown
            process.puJetIdForNoPileUpMEtJetEnDown = process.puJetIdForNoPileUpMEt.clone(
                jetids = cms.InputTag('puJetIdDataForNoPileUpMEtJetEnDown'),
                jets = cms.InputTag('correctedJetsEnDownForNoPileUpPFMEt')
            )
            metUncertaintySequence += process.puJetIdForNoPileUpMEtJetEnDown
            process.noPileUpPFMEtDataJetEnDown = process.noPileUpPFMEtData.clone(
                srcJets = cms.InputTag('correctedJetsEnDownForNoPileUpPFMEt'),
                srcJetIds = cms.InputTag('puJetIdForNoPileUpMEtJetEnDown', 'fullId')
            )
            metUncertaintySequence += process.noPileUpPFMEtDataJetEnDown
            process.noPileUpPFMEtJetEnDown = process.noPileUpPFMEt.clone(
                srcMVAMEtData = cms.InputTag('noPileUpPFMEtDataJetEnDown'),                                           
                srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections))
            )
            metUncertaintySequence += process.noPileUpPFMEtJetEnDown
            self._addPATMEtProducer(process, metUncertaintySequence,
                                    'noPileUpPFMEtJetEnDown', 'patPFMetNoPileUpJetEnDown', collectionsToKeep)

            if hasattr(process, "smearedUncorrectedJetsForNoPileUpPFMEt"):
                process.smearedCorrectedJetsForNoPileUpPFMEt = process.smearedUncorrectedJetsForNoPileUpPFMEt.clone(
                    src = cms.InputTag('calibratedAK5PFJetsForNoPileUpMEt'),
                    jetCorrLabel = cms.string("")
                )
                process.correctedJetsResUpForNoPileUpPFMEt = process.smearedCorrectedJetsForNoPileUpPFMEt.clone(
                    shiftBy = cms.double(-1.*varyByNsigmas)
                )
                metUncertaintySequence += process.correctedJetsResUpForNoPileUpPFMEt
                process.correctedJetsResDownForNoPileUpPFMEt = process.smearedCorrectedJetsForNoPileUpPFMEt.clone(
                    shiftBy = cms.double(+1.*varyByNsigmas)
                )  
                metUncertaintySequence += process.correctedJetsResDownForNoPileUpPFMEt
                process.puJetIdDataForNoPileUpMEtJetResUp = process.puJetIdDataForNoPileUpMEt.clone(
                    jets = cms.InputTag('correctedJetsResUpForNoPileUpPFMEt')
                )
                metUncertaintySequence += process.puJetIdDataForNoPileUpMEtJetResUp
                process.puJetIdForNoPileUpMEtJetResUp = process.puJetIdForNoPileUpMEt.clone(
                    jetids = cms.InputTag('puJetIdDataForNoPileUpMEtJetResUp'),
                    jets = cms.InputTag('correctedJetsResUpForNoPileUpPFMEt')
                )
                metUncertaintySequence += process.puJetIdForNoPileUpMEtJetResUp
                process.noPileUpPFMEtDataJetResUp = process.noPileUpPFMEtData.clone(
                    srcJets = cms.InputTag('correctedJetsResUpForNoPileUpPFMEt'),
                    srcJetIds = cms.InputTag('puJetIdForNoPileUpMEtJetResUp', 'fullId')
                )
                metUncertaintySequence += process.noPileUpPFMEtDataJetResUp
                process.noPileUpPFMEtJetResUp = process.noPileUpPFMEt.clone(
                    srcMVAMEtData = cms.InputTag('noPileUpPFMEtDataJetResUp'),                                           
                    srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections))
                )
                metUncertaintySequence += process.noPileUpPFMEtJetResUp
                self._addPATMEtProducer(process, metUncertaintySequence,
                                        'noPileUpPFMEtJetResUp', 'patPFMetNoPileUpJetResUp', collectionsToKeep)
                process.puJetIdDataForNoPileUpMEtJetResDown = process.puJetIdDataForNoPileUpMEt.clone(
                    jets = cms.InputTag('correctedJetsResDownForNoPileUpPFMEt')
                )
                metUncertaintySequence += process.puJetIdDataForNoPileUpMEtJetResDown
                process.puJetIdForNoPileUpMEtJetResDown = process.puJetIdForNoPileUpMEt.clone(
                    jetids = cms.InputTag('puJetIdDataForNoPileUpMEtJetResDown'),
                    jets = cms.InputTag('correctedJetsResDownForNoPileUpPFMEt')
                )
                metUncertaintySequence += process.puJetIdForNoPileUpMEtJetResDown
                process.noPileUpPFMEtDataJetResDown = process.noPileUpPFMEtData.clone(
                    srcJets = cms.InputTag('correctedJetsResDownForNoPileUpPFMEt'),
                    srcJetIds = cms.InputTag('puJetIdForNoPileUpMEtJetResDown', 'fullId')
                )
                metUncertaintySequence += process.noPileUpPFMEtDataJetResDown
                process.noPileUpPFMEtJetResDown = process.noPileUpPFMEt.clone(
                    srcMVAMEtData = cms.InputTag('noPileUpPFMEtDataJetResDown'),                                           
                    srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections))
                )
                metUncertaintySequence += process.noPileUpPFMEtJetResDown
                self._addPATMEtProducer(process, metUncertaintySequence,
                                        'noPileUpPFMEtJetResDown', 'patPFMetNoPileUpJetResDown', collectionsToKeep)
                        
            process.pfCandsUnclusteredEnUpForNoPileUpPFMEt = cms.EDProducer("ShiftedPFCandidateProducerForNoPileUpPFMEt",
                srcPFCandidates = cms.InputTag('particleFlow'),
                srcJets = cms.InputTag('calibratedAK5PFJetsForNoPileUpMEt'),
                jetCorrInputFileName = cms.FileInPath('PhysicsTools/PatUtils/data/Summer12_V2_DATA_AK5PF_UncertaintySources.txt'),
                jetCorrUncertaintyTag = cms.string("SubTotalDataMC"),
                minJetPt = cms.double(10.0), 
                shiftBy = cms.double(+1.*varyByNsigmas),
                unclEnUncertainty = cms.double(0.10)
            )
            metUncertaintySequence += process.pfCandsUnclusteredEnUpForNoPileUpPFMEt
            process.pfCandidateToVertexAssociationUnclusteredEnUpForNoPileUpPFMEt = process.pfCandidateToVertexAssociation.clone(
                PFCandidateCollection = cms.InputTag('pfCandsUnclusteredEnUpForNoPileUpPFMEt')
            )
            metUncertaintySequence += process.pfCandidateToVertexAssociationUnclusteredEnUpForNoPileUpPFMEt
            process.noPileUpPFMEtDataUnclusteredEnUp = process.noPileUpPFMEtData.clone(              
                srcPFCandidates = cms.InputTag('pfCandsUnclusteredEnUpForNoPileUpPFMEt'),
                srcPFCandToVertexAssociations = cms.InputTag('pfCandidateToVertexAssociationUnclusteredEnUpForNoPileUpPFMEt'),
            )
            metUncertaintySequence += process.noPileUpPFMEtDataUnclusteredEnUp
            process.noPileUpPFMEtUnclusteredEnUp = process.noPileUpPFMEt.clone(
                srcMVAMEtData = cms.InputTag('noPileUpPFMEtDataUnclusteredEnUp'),                                           
                srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections))
            )
            metUncertaintySequence += process.noPileUpPFMEtUnclusteredEnUp
            self._addPATMEtProducer(process, metUncertaintySequence,
                                    'noPileUpPFMEtUnclusteredEnUp', 'patPFMetNoPileUpUnclusteredEnUp', collectionsToKeep)
            process.pfCandsUnclusteredEnDownForNoPileUpPFMEt = process.pfCandsUnclusteredEnUpForNoPileUpPFMEt.clone(
                shiftBy = cms.double(-1.*varyByNsigmas),
            )
            metUncertaintySequence += process.pfCandsUnclusteredEnDownForNoPileUpPFMEt
            process.pfCandidateToVertexAssociationUnclusteredEnDownForNoPileUpPFMEt = process.pfCandidateToVertexAssociation.clone(
                PFCandidateCollection = cms.InputTag('pfCandsUnclusteredEnDownForNoPileUpPFMEt')
            )
            metUncertaintySequence += process.pfCandidateToVertexAssociationUnclusteredEnDownForNoPileUpPFMEt
            process.noPileUpPFMEtDataUnclusteredEnDown = process.noPileUpPFMEtData.clone(              
                srcPFCandidates = cms.InputTag('pfCandsUnclusteredEnDownForNoPileUpPFMEt'),
                srcPFCandToVertexAssociations = cms.InputTag('pfCandidateToVertexAssociationUnclusteredEnDownForNoPileUpPFMEt'),
            )
            metUncertaintySequence += process.noPileUpPFMEtDataUnclusteredEnDown
            process.noPileUpPFMEtUnclusteredEnDown = process.noPileUpPFMEt.clone(
                srcMVAMEtData = cms.InputTag('noPileUpPFMEtDataUnclusteredEnDown'),                                           
                srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections))
            )
            metUncertaintySequence += process.noPileUpPFMEtUnclusteredEnDown
            self._addPATMEtProducer(process, metUncertaintySequence,
                                    'noPileUpPFMEtUnclusteredEnDown', 'patPFMetNoPileUpUnclusteredEnDown', collectionsToKeep)
    
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
                 outputModule            = None):
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

        process.metUncertaintySequence = cms.Sequence()

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
                                        process.metUncertaintySequence)
        cleanedJetCollection = lastJetCollection 
        
        # smear jet energies to account for difference in jet resolutions between MC and Data
        # (cf. JME-10-014 PAS)        
        jetCollectionResUp = None
        jetCollectionResDown = None
        if doSmearJets:
            lastJetCollection = \
              self._addSmearedJets(process, cleanedJetCollection, [ "smeared", jetCollection.value() ],
                                   jetSmearFileName, jetSmearHistogram, varyByNsigmas)
                
            jetCollectionResUp = \
              self._addSmearedJets(process, cleanedJetCollection, [ "smeared", jetCollection.value(), "ResUp" ],
                                   jetSmearFileName, jetSmearHistogram, varyByNsigmas,
                                   -1.)            
            collectionsToKeep.append(jetCollectionResUp)
            jetCollectionResDown = \
              self._addSmearedJets(process, cleanedJetCollection, [ "smeared", jetCollection.value(), "ResDown" ],
                                   jetSmearFileName, jetSmearHistogram, varyByNsigmas,
                                   +1.)                
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
                                              varyByNsigmas)
        process.metUncertaintySequence += process.shiftedParticlesForMEtUncertainties
        collectionsToKeep.extend(addCollectionsToKeep)
        
        #--------------------------------------------------------------------------------------------    
        # propagate shifted particle energies to Type 1 and Type 1 + 2 corrected PFMET
        #--------------------------------------------------------------------------------------------

        self._addCorrPFMEt(process, process.metUncertaintySequence,
                           shiftedParticleCollections, pfCandCollection,
                           collectionsToKeep,
                           doSmearJets,
                           makeType1corrPFMEt,
                           makeType1p2corrPFMEt,
                           doApplyType0corr,
                           sysShiftCorrParameter,                           
                           doApplySysShiftCorr,
                           jetCorrLabel,
                           varyByNsigmas)

        #--------------------------------------------------------------------------------------------    
        # propagate shifted particle energies to MVA-based PFMET
        #--------------------------------------------------------------------------------------------

        self._addPFMEtByMVA(process, process.metUncertaintySequence,
                            shiftedParticleCollections, pfCandCollection,
                            collectionsToKeep,
                            doSmearJets,
                            makePFMEtByMVA,
                            varyByNsigmas)

        #--------------------------------------------------------------------------------------------    
        # propagate shifted particle energies to no-PU PFMET
        #--------------------------------------------------------------------------------------------

        self._addNoPileUpPFMEt(process, process.metUncertaintySequence,
                               shiftedParticleCollections, pfCandCollection,
                               collectionsToKeep,
                               doSmearJets,
                               makeNoPileUpPFMEt,
                               varyByNsigmas)
        
        # insert metUncertaintySequence into patDefaultSequence
        if addToPatDefaultSequence:
            if not hasattr(process, "patDefaultSequence"):
                raise ValueError("PAT default sequence is not defined !!")
            process.patDefaultSequence += process.metUncertaintySequence        
       
        # add shifted + unshifted collections pf pat::Electrons/Photons,
        # Muons, Taus, Jets and MET to PAT-tuple event content
        if outputModule is not None and hasattr(process, outputModule):
            getattr(process, outputModule).outputCommands = _addEventContent(
                getattr(process, outputModule).outputCommands,
                [ 'keep *_%s_*_%s' % (collectionToKeep, process.name_()) for collectionToKeep in collectionsToKeep ])
       
runMEtUncertainties=RunMEtUncertainties()
