import FWCore.ParameterSet.Config as cms

from FWCore.GuiBrowsers.ConfigToolBase import *

import PhysicsTools.PatAlgos.tools.helpers as configtools
from PhysicsTools.PatAlgos.tools.trigTools import _addEventContent

from PhysicsTools.PatUtils.patPFMETCorrections_cff import *
import RecoMET.METProducers.METSigParams_cfi as jetResolutions
 
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
        self.addParameter(self._defaultParameters, 'doApplyType0corr', True, 
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
            dRmaxGenJetMatch = cms.string('TMath::Min(0.5, 0.1 + 0.3*TMath::Exp(-0.05*genJetPt - 10.))'),                         
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
        if input is None or input.value() == '""':
            return False
        else:
            return True
    
    def __call__(self, process,
                 electronCollection      = None,
                 photonCollection        = None,
                 muonCollection          = None,
                 tauCollection           = None,
                 jetCollection           = None,
                 dRjetCleaning           = None,
                 jetCorrLabel            = None,
                 doSmearJets             = None,
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
        # produce collection of jets shifted up/down in energy    
        #--------------------------------------------------------------------------------------------     

        # in case of "raw" (uncorrected) MET,
        # add residual jet energy corrections in quadrature to jet energy uncertainties:
        # cf. https://twiki.cern.ch/twiki/bin/view/CMS/MissingETUncertaintyPrescription
        jetsEnUpForRawMEt = cms.EDProducer("ShiftedPATJetProducer",
            src = cms.InputTag(lastJetCollection),
            #jetCorrPayloadName = cms.string(jetCorrPayloadName),
            #jetCorrUncertaintyTag = cms.string('Uncertainty'),
            jetCorrInputFileName = cms.FileInPath('PhysicsTools/PatUtils/data/JEC11_V12_AK5PF_UncertaintySources.txt'),
            jetCorrUncertaintyTag = cms.string("SubTotalDataMC"),
            addResidualJES = cms.bool(True),
            jetCorrLabelUpToL3 = cms.string("ak5PFL1FastL2L3"),
            jetCorrLabelUpToL3Res = cms.string("ak5PFL1FastL2L3Residual"),                               
            shiftBy = cms.double(+1.*varyByNsigmas)
        )
        jetCollectionEnUpForRawMEt = \
          self._addModuleToSequence(process, jetsEnUpForRawMEt,
                                    [ "shifted", jetCollection.value(), "EnUpForRawMEt" ],
                                    process.metUncertaintySequence)
        collectionsToKeep.append(jetCollectionEnUpForRawMEt)
        jetsEnDownForRawMEt = jetsEnUpForRawMEt.clone(
            shiftBy = cms.double(-1.*varyByNsigmas)
        )
        jetCollectionEnDownForRawMEt = \
          self._addModuleToSequence(process, jetsEnDownForRawMEt,
                                    [ "shifted", jetCollection.value(), "EnDownForRawMEt" ],
                                    process.metUncertaintySequence) 
        collectionsToKeep.append(jetCollectionEnDownForRawMEt)

        jetsEnUpForCorrMEt = jetsEnUpForRawMEt.clone(
            addResidualJES = cms.bool(False)
        )
        jetCollectionEnUpForCorrMEt = \
          self._addModuleToSequence(process, jetsEnUpForCorrMEt,
                                    [ "shifted", jetCollection.value(), "EnUpForCorrMEt" ],
                                    process.metUncertaintySequence)
        collectionsToKeep.append(jetCollectionEnUpForCorrMEt)
        jetsEnDownForCorrMEt = jetsEnUpForCorrMEt.clone(
            shiftBy = cms.double(-1.*varyByNsigmas)
        )
        jetCollectionEnDownForCorrMEt = \
          self._addModuleToSequence(process, jetsEnDownForCorrMEt,
                                    [ "shifted", jetCollection.value(), "EnDownForCorrMEt" ],
                                    process.metUncertaintySequence) 
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
                                        process.metUncertaintySequence)
            collectionsToKeep.append(electronCollectionEnUp)
            electronsEnDown = electronsEnUp.clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            )
            electronCollectionEnDown = \
              self._addModuleToSequence(process, electronsEnDown,
                                        [ "shifted", electronCollection.value(), "EnDown" ],
                                        process.metUncertaintySequence)
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
                                        process.metUncertaintySequence)
            collectionsToKeep.append(photonCollectionEnUp)
            photonsEnDown = photonsEnUp.clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            )
            photonCollectionEnDown = \
              self._addModuleToSequence(process, photonsEnDown,
                                        [ "shifted", photonCollection.value(), "EnDown" ],
                                        process.metUncertaintySequence)
            collectionsToKeep.append(photonCollectionEnDown)

        #--------------------------------------------------------------------------------------------
        # produce collection of muons shifted up/down in energy/momentum  
        #--------------------------------------------------------------------------------------------

        muonCollectionEnUp = None
        muonCollectionEnDown = None   
        if self._isValidInputTag(muonCollection):
            muonsEnUp = cms.EDProducer("ShiftedPATMuonProducer",
                src = muonCollection,
                uncertainty = cms.double(0.01),
                shiftBy = cms.double(+1.*varyByNsigmas)
            )
            muonCollectionEnUp = \
              self._addModuleToSequence(process, muonsEnUp,
                                        [ "shifted", muonCollection.value(), "EnUp" ],
                                        process.metUncertaintySequence)
            collectionsToKeep.append(muonCollectionEnUp)
            muonsEnDown = muonsEnUp.clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            )
            muonCollectionEnDown = \
              self._addModuleToSequence(process, muonsEnDown,
                                        [ "shifted", muonCollection.value(), "EnDown" ],
                                        process.metUncertaintySequence)
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
                                        process.metUncertaintySequence)
            collectionsToKeep.append(tauCollectionEnUp)
            tausEnDown = tausEnUp.clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            )
            tauCollectionEnDown = \
              self._addModuleToSequence(process, tausEnDown,
                                        [ "shifted", tauCollection.value(), "EnDown" ],
                                        process.metUncertaintySequence)     
            collectionsToKeep.append(tauCollectionEnDown)

        #--------------------------------------------------------------------------------------------    
        # propagate shifted jet energies to MET
        #--------------------------------------------------------------------------------------------

        # add "nominal" (unshifted) pat::MET collections        
        process.pfCandsNotInJet.bottomCollection = pfCandCollection        
        process.selectedPatJetsForMETtype1p2Corr.src = lastJetCollection
        process.selectedPatJetsForMETtype2Corr.src = lastJetCollection
        
        if not hasattr(process, 'producePatPFMETCorrections'):
            process.load("PhysicsTools.PatUtils.patPFMETCorrections_cff")

        if doApplySysShiftCorr:
            if not hasattr(process, 'pfMEtSysShiftCorrSequence'):
                process.load("JetMETCorrections.Type1MET.pfMETsysShiftCorrections_cfi")
            process.pfMEtSysShiftCorr.parameter = sysShiftCorrParameter
            process.metUncertaintySequence += process.pfMEtSysShiftCorrSequence

        process.metUncertaintySequence += process.producePatPFMETCorrections
        
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

        process.selectedPatJetsForMETtype1p2CorrEnUp = getattr(process, jetCollectionEnUpForCorrMEt).clone(
            src = cms.InputTag('selectedPatJetsForMETtype1p2Corr')
        )
        process.metUncertaintySequence += process.selectedPatJetsForMETtype1p2CorrEnUp
        process.selectedPatJetsForMETtype2CorrEnUp = getattr(process, jetCollectionEnUpForCorrMEt).clone(
            src = cms.InputTag('selectedPatJetsForMETtype2Corr')
        )
        process.metUncertaintySequence += process.selectedPatJetsForMETtype2CorrEnUp
        process.selectedPatJetsForMETtype1p2CorrEnDown = getattr(process, jetCollectionEnDownForCorrMEt).clone(
            src = cms.InputTag('selectedPatJetsForMETtype1p2Corr')
        )
        process.metUncertaintySequence += process.selectedPatJetsForMETtype1p2CorrEnDown
        process.selectedPatJetsForMETtype2CorrEnDown = getattr(process, jetCollectionEnDownForCorrMEt).clone(
            src = cms.InputTag('selectedPatJetsForMETtype2Corr')
        )
        process.metUncertaintySequence += process.selectedPatJetsForMETtype2CorrEnDown    

        if doSmearJets:
            process.selectedPatJetsForMETtype1p2CorrResUp = getattr(process, jetCollectionResUp).clone(
                src = cms.InputTag('selectedPatJetsForMETtype1p2Corr')
            )
            process.metUncertaintySequence += process.selectedPatJetsForMETtype1p2CorrResUp
            process.selectedPatJetsForMETtype2CorrResUp = getattr(process, jetCollectionResUp).clone(
                src = cms.InputTag('selectedPatJetsForMETtype2Corr')
            )
            process.metUncertaintySequence += process.selectedPatJetsForMETtype2CorrResUp
            process.selectedPatJetsForMETtype1p2CorrResDown = getattr(process, jetCollectionResDown).clone(
                src = cms.InputTag('selectedPatJetsForMETtype1p2Corr')
            )
            process.metUncertaintySequence += process.selectedPatJetsForMETtype1p2CorrResDown
            process.selectedPatJetsForMETtype2CorrResDown = getattr(process, jetCollectionResDown).clone(
                src = cms.InputTag('selectedPatJetsForMETtype2Corr')
            )
            process.metUncertaintySequence += process.selectedPatJetsForMETtype2CorrResDown

        if doSmearJets:
            # apply MET smearing to "raw" (uncorrected) MET
            process.smearedPatPFMetSequence = cms.Sequence()
            process.patPFMetForMEtUncertainty = process.patPFMet.clone()
            process.smearedPatPFMetSequence += process.patPFMetForMEtUncertainty
            process.patPFMETcorrJetSmearing = cms.EDProducer("ShiftedParticleMETcorrInputProducer",
                srcOriginal = cms.InputTag(cleanedJetCollection),
                srcShifted = cms.InputTag(lastJetCollection)                                           
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
            process.metUncertaintySequence += process.smearedPatPFMetSequence 

        # propagate shifts in jet energy to "raw" (uncorrected) and Type 1 corrected MET
        metCollectionsUp_DownForRawMEt = \
            self._propagateMEtUncertainties(
                process, lastJetCollection, "Jet", "En", jetCollectionEnUpForRawMEt, jetCollectionEnDownForRawMEt,
                process.patPFMet, process.metUncertaintySequence)
        collectionsToKeep.extend(metCollectionsUp_DownForRawMEt)

        metCollectionsUp_DownForCorrMEt = \
            self._propagateMEtUncertainties(
                process, lastJetCollection, "Jet", "En", jetCollectionEnUpForCorrMEt, jetCollectionEnDownForCorrMEt,
                process.patType1CorrectedPFMet, process.metUncertaintySequence)
        collectionsToKeep.extend(metCollectionsUp_DownForCorrMEt)

        # propagate shifts in jet energy to Type 1 + 2 corrected MET
        process.patPFJetMETtype1p2CorrEnUp = process.patPFJetMETtype1p2Corr.clone(
            src = cms.InputTag(process.selectedPatJetsForMETtype1p2CorrEnUp.label()),
            jetCorrLabel = cms.string(jetCorrLabel)
        )
        process.metUncertaintySequence += process.patPFJetMETtype1p2CorrEnUp
        process.patPFJetMETtype2CorrEnUp = process.patPFJetMETtype2Corr.clone(
            src = cms.InputTag('selectedPatJetsForMETtype2CorrEnUp')
        )
        process.metUncertaintySequence += process.patPFJetMETtype2CorrEnUp
        process.patPFJetMETtype1p2CorrEnDown = process.patPFJetMETtype1p2CorrEnUp.clone(
            src = cms.InputTag(process.selectedPatJetsForMETtype1p2CorrEnDown.label())
        )
        process.metUncertaintySequence += process.patPFJetMETtype1p2CorrEnDown
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
        process.metUncertaintySequence += process.patType1p2CorrectedPFMetJetEnUp
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
        process.metUncertaintySequence += process.patType1p2CorrectedPFMetJetEnDown
        collectionsToKeep.append('patType1p2CorrectedPFMetJetEnDown')

        if doSmearJets:
            # propagate shifts in jet resolution to "raw" (uncorrected) MET and Type 1 corrected MET
            for metProducer in [ process.patPFMet,
                                 process.patType1CorrectedPFMet ]:

                metCollectionsUp_Down = \
                    self._propagateMEtUncertainties(
                        process, lastJetCollection, "Jet", "Res", jetCollectionResUp, jetCollectionResDown,
                        metProducer, process.metUncertaintySequence)
                collectionsToKeep.extend(metCollectionsUp_Down)
            
            # propagate shifts in jet resolution to Type 1 + 2 corrected MET 
            process.patPFJetMETtype1p2CorrResUp = process.patPFJetMETtype1p2Corr.clone(
                src = cms.InputTag(process.selectedPatJetsForMETtype1p2CorrResUp.label()),
                jetCorrLabel = cms.string(jetCorrLabel)
            )
            process.metUncertaintySequence += process.patPFJetMETtype1p2CorrResUp
            process.patPFJetMETtype2CorrResUp = process.patPFJetMETtype2Corr.clone(
                src = cms.InputTag('selectedPatJetsForMETtype2CorrResUp')
            )
            process.metUncertaintySequence += process.patPFJetMETtype2CorrResUp
            process.patPFJetMETtype1p2CorrResDown = process.patPFJetMETtype1p2CorrResUp.clone(
                src = cms.InputTag(process.selectedPatJetsForMETtype1p2CorrResDown.label())
            )
            process.metUncertaintySequence += process.patPFJetMETtype1p2CorrResDown
            process.patPFJetMETtype2CorrResDown = process.patPFJetMETtype2Corr.clone(
                src = cms.InputTag('selectedPatJetsForMETtype2CorrResDown')
            )
            process.metUncertaintySequence += process.patPFJetMETtype2CorrResDown

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
            process.metUncertaintySequence += process.patType1p2CorrectedPFMetJetResUp
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
            process.metUncertaintySequence += process.patType1p2CorrectedPFMetJetResDown
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
        process.metUncertaintySequence += process.patType1p2CorrectedPFMetUnclusteredEnUp
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
        process.metUncertaintySequence += process.patType1p2CorrectedPFMetUnclusteredEnDown
        collectionsToKeep.append('patType1p2CorrectedPFMetUnclusteredEnDown')

        #--------------------------------------------------------------------------------------------    
        # propagate shifted electron/photon, muon and tau-jet energies to MET
        #--------------------------------------------------------------------------------------------

        for metProducer in [ process.patPFMet,
                             process.patType1CorrectedPFMet,
                             process.patType1p2CorrectedPFMet ]:
            
            if self._isValidInputTag(electronCollection):
                metCollectionsUp_Down = \
                    self._propagateMEtUncertainties(
                        process, electronCollection.value(), "Electron", "En", electronCollectionEnUp, electronCollectionEnDown,
                        metProducer, process.metUncertaintySequence)
                collectionsToKeep.extend(metCollectionsUp_Down)

            if self._isValidInputTag(photonCollection):
                metCollectionsUp_Down = \
                    self._propagateMEtUncertainties(
                        process, photonCollection.value(), "Photon", "En", photonCollectionEnUp, photonCollectionEnDown,
                        metProducer, process.metUncertaintySequence)
                collectionsToKeep.extend(metCollectionsUp_Down)
                
            if self._isValidInputTag(muonCollection):
                metCollectionsUp_Down = \
                    self._propagateMEtUncertainties(
                        process, muonCollection.value(), "Muon", "En", muonCollectionEnUp, muonCollectionEnDown,
                        metProducer, process.metUncertaintySequence)
                collectionsToKeep.extend(metCollectionsUp_Down)

            if self._isValidInputTag(tauCollection):
                metCollectionsUp_Down = \
                    self._propagateMEtUncertainties(
                        process, tauCollection.value(), "Tau", "En", tauCollectionEnUp, tauCollectionEnDown,
                        metProducer, process.metUncertaintySequence)
                collectionsToKeep.extend(metCollectionsUp_Down)

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
