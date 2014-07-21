import FWCore.ParameterSet.Config as cms

from FWCore.GuiBrowsers.ConfigToolBase import *

import PhysicsTools.PatAlgos.tools.helpers as configtools
from PhysicsTools.PatAlgos.tools.trigTools import _addEventContent
from PhysicsTools.PatUtils.tools.jmeUncertaintyTools import JetMEtUncertaintyTools

from PhysicsTools.PatUtils.patPFMETCorrections_cff import *
import RecoMET.METProducers.METSigParams_cfi as jetResolutions
from PhysicsTools.PatAlgos.producersLayer1.metProducer_cfi import patMETs


class RunType1PFMEtUncertainties(JetMEtUncertaintyTools):

    """ Shift energy of electrons, photons, muons, tau-jets and other jets
    reconstructed in the event up/down,
    in order to estimate effect of energy scale uncertainties on Type-1 corrected PFMET
   """
    _label='runType1PFMEtUncertainties'
    _defaultParameters = dicttypes.SortedKeysDict()
    def __init__(self):
        JetMEtUncertaintyTools.__init__(self)
	self.addParameter(self._defaultParameters, 'dRjetCleaning', 0.5, 
                          "Eta-phi distance for extra jet cleaning", Type=float)
        self.addParameter(self._defaultParameters, 'makeType1corrPFMEt', True,
                          "Flag to enable/disable sequence for Type 1 corrected PFMEt", Type=bool)
        self.addParameter(self._defaultParameters, 'makeType1p2corrPFMEt', True,
                          "Flag to enable/disable sequence for Type 1 + 2 corrected PFMEt", Type=bool)
        self.addParameter(self._defaultParameters, 'doApplyType0corr', False,
                          "Flag to enable/disable usage of Type-0 MET corrections", Type=bool)
        self.addParameter(self._defaultParameters, 'sysShiftCorrParameter', cms.VPSet(),
                          "MET sys. shift correction parameters", Type=cms.VPSet)
        self.addParameter(self._defaultParameters, 'doApplySysShiftCorr', False,
                          "Flag to enable/disable usage of MET sys. shift corrections", Type=bool)
	self.addParameter(self._defaultParameters, 'pfCandCollection', cms.InputTag('particleFlow'), 
                          "Input PFCandidate collection", Type=cms.InputTag)
        self.addParameter(self._defaultParameters, 'doApplyUnclEnergyCalibration', False,
                          "Flag to enable/disable usage of 'unclustered energy' calibration", Type=bool)
        self._parameters = copy.deepcopy(self._defaultParameters)
        self._comment = ""
   
    def _addCorrPFMEt(self, process, metUncertaintySequence,
                      shiftedParticleCollections, pfCandCollection, doApplyUnclEnergyCalibration,
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


        ## standard naming convention
        metModNameT1="patPFMet"
        if doApplyType0corr :
            metModNameT1 += "T0pc"
        metModNameT1 += "T1"
        if doApplySysShiftCorr :
            metModNameT1 += "Txy"
       
        metModNameT1T2="patPFMet"
        if doApplyType0corr :
            metModNameT1T2 += "T0pc"
        metModNameT1T2 += "T1T2"
        if doApplySysShiftCorr :
            metModNameT1T2 += "Txy"
        ## MM

        if not (makeType1corrPFMEt or makeType1p2corrPFMEt):
            return

        if not hasattr(process, 'producePatPFMETCorrectionsUnc'):
            process.load("PhysicsTools.PatUtils.patPFMETCorrections_cff")
        
        #adapt modules to the good naming scheme
        process.producePatPFMETCorrectionsUnc += getattr(process, metModNameT1 )
        process.producePatPFMETCorrectionsUnc += getattr(process, metModNameT1T2 )
       
        # If with empty postfix, make a backup of
        # process.producePatPFMETCorrectionsUnc, because the original
        # sequence will be modified later in this function
        if postfix == "":
            configtools.cloneProcessingSnippet(process, process.producePatPFMETCorrectionsUnc, "OriginalReserved")
        else:
            if postfix == "OriginalReserved":
                raise ValueError("Postfix label '%s' is reserved for internal usage !!" % postfix)

            if hasattr(process, "producePatPFMETCorrectionsUncOriginalReserved"):
                configtools.cloneProcessingSnippet(process, process.producePatPFMETCorrectionsUncOriginalReserved, postfix, removePostfix="OriginalReserved")
            else:
                configtools.cloneProcessingSnippet(process, process.producePatPFMETCorrectionsUnc, postfix)

        calibratedPFCandsNotInJets = None
        if doApplyUnclEnergyCalibration:
            patPFJetMETtype1p2Corr = getattr(process, "patPFJetMETtype1p2Corr" + postfix)
            patPFJetMETtype1p2Corr.type2ResidualCorrLabel = cms.string("")
            patPFJetMETtype1p2Corr.type2ResidualCorrEtaMax = cms.double(9.9)
            patPFJetMETtype1p2Corr.type2ResidualCorrOffset = cms.double(1.)
            patPFJetMETtype1p2Corr.type2ExtraCorrFactor = cms.double(1.)
            patPFJetMETtype1p2Corr.isMC = cms.bool(True)
            patPFJetMETtype1p2Corr.srcGenPileUpSummary = cms.InputTag('addPileupInfo')
            patPFJetMETtype1p2Corr.type2ResidualCorrVsNumPileUp = cms.PSet(
                data = cms.PSet(
                    offset = cms.FileInPath('JetMETCorrections/Type1MET/data/unclEnResidualCorr_Data_runs190456to208686_pfCands_offset.txt'),
                    slope = cms.FileInPath('JetMETCorrections/Type1MET/data/unclEnResidualCorr_Data_runs190456to208686_pfCands_slope.txt')
                ),
                mc = cms.PSet(
                    offset = cms.FileInPath('JetMETCorrections/Type1MET/data/unclEnResidualCorr_ZplusJets_madgraph_pfCands_offset.txt'),
                    slope = cms.FileInPath('JetMETCorrections/Type1MET/data/unclEnResidualCorr_ZplusJets_madgraph_pfCands_slope.txt')
                )
            )
            patPFJetMETtype1p2Corr.verbosity = cms.int32(0)
            pfCandMETcorr = getattr(process, "pfCandMETcorr" + postfix)
            pfCandMETresidualCorr = pfCandMETcorr.clone(
                residualCorrLabel = cms.string(""),
                residualCorrEtaMax = cms.double(9.9),
                residualCorrOffset = cms.double(1.),
                extraCorrFactor = cms.double(1.),
                isMC = cms.bool(True),
                srcGenPileUpSummary = cms.InputTag('addPileupInfo'),
                residualCorrVsNumPileUp = cms.PSet(
                    data = cms.PSet(
                        offset = cms.FileInPath('JetMETCorrections/Type1MET/data/unclEnResidualCorr_Data_runs190456to208686_pfCands_offset.txt'),
                        slope = cms.FileInPath('JetMETCorrections/Type1MET/data/unclEnResidualCorr_Data_runs190456to208686_pfCands_slope.txt')
                    ),
                    mc = cms.PSet(
                        offset = cms.FileInPath('JetMETCorrections/Type1MET/data/unclEnResidualCorr_ZplusJets_madgraph_pfCands_offset.txt'),
                        slope = cms.FileInPath('JetMETCorrections/Type1MET/data/unclEnResidualCorr_ZplusJets_madgraph_pfCands_slope.txt')
                    )
                ),
                verbosity = cms.int32(0)  
            )
            setattr(process, "pfCandMETresidualCorr" + postfix, pfCandMETresidualCorr)
            getattr(process, "producePatPFMETCorrectionsUnc" + postfix).replace(pfCandMETcorr, pfCandMETcorr + pfCandMETresidualCorr)

            patPFMetT1 = getattr(process, metModNameT1 + postfix)
            patPFMetT1.applyType2Corrections = cms.bool(True)
            patPFMetT1.srcUnclEnergySums = cms.VInputTag(
                cms.InputTag('pfCandMETresidualCorr' + postfix),
                cms.InputTag("patPFJetMETtype1p2Corr" + postfix, "type2")
            )
            patPFMetT1.type2CorrFormula = cms.string("A")
            patPFMetT1.type2CorrParameter = cms.PSet(A = cms.double(2.))

        ##getattr(process, "patPFJetMETtype1p2Corr" + postfix).verbosity = cms.int32(1)  
        ##getattr(process, "patPFMetT1" + postfix).verbosity = cms.int32(1)  
        
        # add "nominal" (unshifted) pat::MET collections        
        getattr(process, "pfCandsNotInJet" + postfix).bottomCollection = pfCandCollection
        getattr(process, "selectedPatJetsForMETtype1p2Corr" + postfix).src = shiftedParticleCollections['lastJetCollection']
        getattr(process, "selectedPatJetsForMETtype2Corr" + postfix).src = shiftedParticleCollections['lastJetCollection']
        
        if doApplySysShiftCorr:
            if not hasattr(process, 'pfMEtSysShiftCorrSequence'):
                process.load("JetMETCorrections.Type1MET.pfMETsysShiftCorrections_cfi")
            if postfix != "":
                configtools.cloneProcessingSnippet(process, process.pfMEtSysShiftCorrSequence, postfix)

            getattr(process, "pfMEtSysShiftCorr" + postfix).parameter = sysShiftCorrParameter
            metUncertaintySequence += getattr(process, "pfMEtSysShiftCorrSequence" + postfix)

        metUncertaintySequence += getattr(process, "producePatPFMETCorrectionsUnc" + postfix)
        
        patType1correctionsCentralValue = [ cms.InputTag('patPFJetMETtype1p2Corr' + postfix, 'type1') ]
        if doApplyType0corr:
            patType1correctionsCentralValue.extend([ cms.InputTag('patPFMETtype0Corr' + postfix) ])
        if doApplySysShiftCorr:
            patType1correctionsCentralValue.extend([ cms.InputTag('pfMEtSysShiftCorr' + postfix) ])
        getattr(process, metModNameT1 + postfix).srcType1Corrections = cms.VInputTag(patType1correctionsCentralValue)
        getattr(process, metModNameT1T2 + postfix).srcType1Corrections = cms.VInputTag(patType1correctionsCentralValue)
        
        collectionsToKeep.extend([
            'patPFMet' + postfix,
            metModNameT1 + postfix,
            metModNameT1T2 + postfix])

        setattr(process, "selectedPatJetsForMETtype1p2CorrEnUp" + postfix, 
          getattr(process, shiftedParticleCollections['jetCollectionEnUpForCorrMEt']).clone(
            src = cms.InputTag('selectedPatJetsForMETtype1p2Corr' + postfix)
        ))
        metUncertaintySequence += getattr(process, "selectedPatJetsForMETtype1p2CorrEnUp" + postfix)
        setattr(process, "selectedPatJetsForMETtype1p2CorrEnDown" + postfix,
          getattr(process, shiftedParticleCollections['jetCollectionEnDownForCorrMEt']).clone(
            src = cms.InputTag('selectedPatJetsForMETtype1p2Corr' + postfix)
        ))
        metUncertaintySequence += getattr(process, "selectedPatJetsForMETtype1p2CorrEnDown" + postfix)
        if makeType1p2corrPFMEt:
            setattr(process, "selectedPatJetsForMETtype2CorrEnUp" + postfix,
              getattr(process, shiftedParticleCollections['jetCollectionEnUpForCorrMEt']).clone(
                src = cms.InputTag('selectedPatJetsForMETtype2Corr' + postfix)
            ))
            metUncertaintySequence += getattr(process, "selectedPatJetsForMETtype2CorrEnUp" + postfix)
            setattr(process, "selectedPatJetsForMETtype2CorrEnDown" + postfix,
              getattr(process, shiftedParticleCollections['jetCollectionEnDownForCorrMEt']).clone(
                src = cms.InputTag('selectedPatJetsForMETtype2Corr' + postfix)
            ))
            metUncertaintySequence += getattr(process, "selectedPatJetsForMETtype2CorrEnDown" + postfix)

        if doSmearJets:
            setattr(process, "selectedPatJetsForMETtype1p2CorrResUp" + postfix,
              getattr(process, shiftedParticleCollections['jetCollectionResUp']).clone(
                src = cms.InputTag('selectedPatJetsForMETtype1p2Corr' + postfix)
            ))
            metUncertaintySequence += getattr(process, "selectedPatJetsForMETtype1p2CorrResUp" + postfix)
            setattr(process, "selectedPatJetsForMETtype1p2CorrResDown" + postfix,
              getattr(process, shiftedParticleCollections['jetCollectionResDown']).clone(
                src = cms.InputTag('selectedPatJetsForMETtype1p2Corr' + postfix)
            ))
            metUncertaintySequence += getattr(process, "selectedPatJetsForMETtype1p2CorrResDown" + postfix)
            if makeType1p2corrPFMEt:            
                setattr(process, "selectedPatJetsForMETtype2CorrResUp" + postfix,
                  getattr(process, shiftedParticleCollections['jetCollectionResUp']).clone(
                    src = cms.InputTag('selectedPatJetsForMETtype2Corr' + postfix)
                ))
                metUncertaintySequence += getattr(process, "selectedPatJetsForMETtype2CorrResUp" + postfix)
                setattr(process, "selectedPatJetsForMETtype2CorrResDown" + postfix,
                  getattr(process, shiftedParticleCollections['jetCollectionResDown']).clone(
                    src = cms.InputTag('selectedPatJetsForMETtype2Corr' + postfix)
                ))
                metUncertaintySequence += getattr(process, "selectedPatJetsForMETtype2CorrResDown" + postfix)

        if doSmearJets:
            # apply MET smearing to "raw" (uncorrected) MET
            smearedPatPFMetSequence = cms.Sequence()
            setattr(process, "smearedPatPFMetSequence" + postfix, smearedPatPFMetSequence)
            if not hasattr(process, "patPFMetORIGINAL"):
                setattr(process, "patPFMetORIGINAL" + postfix, getattr(process, "patPFMet").clone())
            setattr(process, "patPFMetForMEtUncertainty" + postfix, getattr(process, "patPFMetORIGINAL" + postfix).clone())
            smearedPatPFMetSequence += getattr(process, "patPFMetForMEtUncertainty" + postfix)
            setattr(process, "patPFMETcorrJetSmearing" + postfix, cms.EDProducer("ShiftedParticleMETcorrInputProducer",
                srcOriginal = cms.InputTag(shiftedParticleCollections['cleanedJetCollection']),
                srcShifted = cms.InputTag(shiftedParticleCollections['lastJetCollection'])                                           
            ))
            smearedPatPFMetSequence += getattr(process, "patPFMETcorrJetSmearing" + postfix)
            getattr(process, "producePatPFMETCorrectionsUnc" + postfix).replace(getattr(process, "patPFMet" + postfix), smearedPatPFMetSequence)
            setattr(process, "patPFMet" + postfix, getattr(process, metModNameT1 + postfix).clone(
                src = cms.InputTag('patPFMetForMEtUncertainty' + postfix),                
                srcType1Corrections = cms.VInputTag(
                    cms.InputTag('patPFMETcorrJetSmearing' + postfix)
                ),
                applyType2Corrections = cms.bool(False),
                srcUnclEnergySums = cms.VInputTag()
            ))
            smearedPatPFMetSequence += getattr(process, "patPFMet" + postfix)
            metUncertaintySequence += smearedPatPFMetSequence 

        # propagate shifts in jet energy to "raw" (uncorrected) and Type 1 corrected MET
        metCollectionsUp_DownForRawMEt = \
            self._propagateMEtUncertainties(
              process, shiftedParticleCollections['lastJetCollection'], "Jet", "En",
              shiftedParticleCollections['jetCollectionEnUpForRawMEt'], shiftedParticleCollections['jetCollectionEnDownForRawMEt'],
              getattr(process, "patPFMet" + postfix), "PF", metUncertaintySequence, postfix)
        collectionsToKeep.extend(metCollectionsUp_DownForRawMEt)

        metCollectionsUp_DownForCorrMEt = \
            self._propagateMEtUncertainties(
              process, shiftedParticleCollections['lastJetCollection'], "Jet", "En",
              shiftedParticleCollections['jetCollectionEnUpForCorrMEt'], shiftedParticleCollections['jetCollectionEnDownForCorrMEt'],
              getattr(process, metModNameT1 + postfix), "PF", metUncertaintySequence, postfix)
        collectionsToKeep.extend(metCollectionsUp_DownForCorrMEt)

        # propagate shifts in jet energy to Type 1 + 2 corrected MET
        if makeType1p2corrPFMEt:   
            setattr(process, "patPFJetMETtype1p2CorrEnUp" + postfix, getattr(process, "patPFJetMETtype1p2Corr" + postfix).clone(
                src = cms.InputTag(getattr(process, "selectedPatJetsForMETtype1p2CorrEnUp" + postfix).label()),
                jetCorrLabel = cms.string(jetCorrLabel)
            ))
            metUncertaintySequence += getattr(process, "patPFJetMETtype1p2CorrEnUp" + postfix)
            setattr(process, "patPFJetMETtype1p2CorrEnDown" + postfix, getattr(process, "patPFJetMETtype1p2CorrEnUp" + postfix).clone(
                src = cms.InputTag(getattr(process, "selectedPatJetsForMETtype1p2CorrEnDown" + postfix).label())
            ))
            metUncertaintySequence += getattr(process, "patPFJetMETtype1p2CorrEnDown" + postfix)
            setattr(process, "patPFJetMETtype2CorrEnUp" + postfix, getattr(process, "patPFJetMETtype2Corr" + postfix).clone(
                src = cms.InputTag('selectedPatJetsForMETtype2CorrEnUp' + postfix)
            ))
            metUncertaintySequence += getattr(process, "patPFJetMETtype2CorrEnUp" + postfix)
            setattr(process, "patPFJetMETtype2CorrEnDown" + postfix,  getattr(process, "patPFJetMETtype2Corr" + postfix).clone(
                src = cms.InputTag('selectedPatJetsForMETtype2CorrEnDown' + postfix)
            ))
            metUncertaintySequence += getattr(process, "patPFJetMETtype2CorrEnDown" + postfix)

            patType1correctionsJetEnUp = [ cms.InputTag('patPFJetMETtype1p2CorrEnUp' + postfix, 'type1') ]        
            if doApplyType0corr:
                patType1correctionsJetEnUp.extend([ cms.InputTag('patPFMETtype0Corr' + postfix) ])
            if doApplySysShiftCorr:
                patType1correctionsJetEnUp.extend([ cms.InputTag('pfMEtSysShiftCorr' + postfix) ])            
            setattr(process, "patPFMetT1T2JetEnUp" + postfix, getattr(process, metModNameT1T2 + postfix).clone(
                srcType1Corrections = cms.VInputTag(patType1correctionsJetEnUp),
                srcUnclEnergySums = cms.VInputTag(
                    cms.InputTag('patPFJetMETtype1p2CorrEnUp' + postfix, 'type2' ),
                    cms.InputTag('patPFJetMETtype2CorrEnUp' + postfix,   'type2' ),
                    cms.InputTag('patPFJetMETtype1p2CorrEnUp' + postfix, 'offset'),
                    cms.InputTag('pfCandMETcorr' + postfix)
                ),                
                applyType2Corrections = cms.bool(True),
                type2CorrParameter = cms.PSet(
                    A = cms.double(1.4)
                )
            ))
            metUncertaintySequence += getattr(process, "patPFMetT1T2JetEnUp" + postfix)
            collectionsToKeep.append('patPFMetT1T2JetEnUp' + postfix)
            patType1correctionsJetEnDown = [ cms.InputTag('patPFJetMETtype1p2CorrEnDown' + postfix, 'type1') ]
            if doApplyType0corr:
                patType1correctionsJetEnDown.extend([ cms.InputTag('patPFMETtype0Corr' + postfix) ])
            if doApplySysShiftCorr:
                patType1correctionsJetEnDown.extend([ cms.InputTag('pfMEtSysShiftCorr' + postfix) ])    
            setattr(process, "patPFMetT1T2JetEnDown" + postfix, getattr(process, "patPFMetT1T2JetEnUp" + postfix).clone(
                srcType1Corrections = cms.VInputTag(patType1correctionsJetEnDown),
                srcUnclEnergySums = cms.VInputTag(
                    cms.InputTag('patPFJetMETtype1p2CorrEnDown' + postfix, 'type2' ),
                    cms.InputTag('patPFJetMETtype2CorrEnDown' + postfix,   'type2' ),
                    cms.InputTag('patPFJetMETtype1p2CorrEnDown' + postfix, 'offset'),
                    cms.InputTag('pfCandMETcorr' + postfix)
                )
            ))
            metUncertaintySequence += getattr(process, "patPFMetT1T2JetEnDown" + postfix)
            collectionsToKeep.append('patPFMetT1T2JetEnDown' + postfix)

        if doSmearJets:
            # propagate shifts in jet resolution to "raw" (uncorrected) MET and Type 1 corrected MET
            for metProducer in [ getattr(process, "patPFMet" + postfix),
                                 getattr(process, metModNameT1 + postfix) ]:

                metCollectionsUp_Down = \
                    self._propagateMEtUncertainties(
                      process, shiftedParticleCollections['lastJetCollection'], "Jet", "Res",
                      shiftedParticleCollections['jetCollectionResUp'], shiftedParticleCollections['jetCollectionResDown'],
                      metProducer, "PF", metUncertaintySequence, postfix)
                collectionsToKeep.extend(metCollectionsUp_Down)
            
            # propagate shifts in jet resolution to Type 1 + 2 corrected MET
            if makeType1p2corrPFMEt:  
                setattr(process, "patPFJetMETtype1p2CorrResUp" + postfix, getattr(process, "patPFJetMETtype1p2Corr" + postfix).clone(
                    src = cms.InputTag(getattr(process, "selectedPatJetsForMETtype1p2CorrResUp" + postfix).label()),
                    jetCorrLabel = cms.string(jetCorrLabel)
                ))
                metUncertaintySequence += getattr(process, "patPFJetMETtype1p2CorrResUp" + postfix)
                setattr(process, "patPFJetMETtype1p2CorrResDown" + postfix, getattr(process, "patPFJetMETtype1p2CorrResUp" + postfix).clone(
                    src = cms.InputTag(getattr(process, "selectedPatJetsForMETtype1p2CorrResDown" + postfix).label())
                ))
                metUncertaintySequence += getattr(process, "patPFJetMETtype1p2CorrResDown" + postfix)
                setattr(process, "patPFJetMETtype2CorrResUp" + postfix, getattr(process, "patPFJetMETtype2Corr" + postfix).clone(
                    src = cms.InputTag('selectedPatJetsForMETtype2CorrResUp' + postfix)
                ))
                metUncertaintySequence += getattr(process, "patPFJetMETtype2CorrResUp" + postfix)
                setattr(process, "patPFJetMETtype2CorrResDown" + postfix, getattr(process, "patPFJetMETtype2Corr" + postfix).clone(
                    src = cms.InputTag('selectedPatJetsForMETtype2CorrResDown' + postfix)
                ))
                metUncertaintySequence += getattr(process, "patPFJetMETtype2CorrResDown" + postfix)

                patType1correctionsJetResUp = [ cms.InputTag('patPFJetMETtype1p2CorrResUp' + postfix, 'type1') ]
                if doApplyType0corr:
                    patType1correctionsJetResUp.extend([ cms.InputTag('patPFMETtype0Corr' + postfix) ])
                if doApplySysShiftCorr:
                    patType1correctionsJetResUp.extend([ cms.InputTag('pfMEtSysShiftCorr' + postfix) ])
                setattr(process, "patPFMetT1T2JetResUp" + postfix,  getattr(process, metModNameT1T2 + postfix).clone(
                    srcType1Corrections = cms.VInputTag(patType1correctionsJetResUp),
                    srcUnclEnergySums = cms.VInputTag(
                        cms.InputTag('patPFJetMETtype1p2CorrResUp' + postfix, 'type2' ),
                        cms.InputTag('patPFJetMETtype2CorrResUp' + postfix,   'type2' ),
                        cms.InputTag('patPFJetMETtype1p2CorrResUp' + postfix, 'offset'),
                        cms.InputTag('pfCandMETcorr' + postfix)
                    ),
                    applyType2Corrections = cms.bool(True),
                    type2CorrParameter = cms.PSet(
                        A = cms.double(1.4)
                    )
                ))
                metUncertaintySequence += getattr(process, "patPFMetT1T2JetResUp" + postfix)
                collectionsToKeep.append('patPFMetT1T2JetResUp' + postfix)
                patType1correctionsJetResDown = [ cms.InputTag('patPFJetMETtype1p2CorrResDown' + postfix, 'type1') ]
                if doApplyType0corr:
                    patType1correctionsJetResDown.extend([ cms.InputTag('patPFMETtype0Corr' + postfix) ])
                if doApplySysShiftCorr:
                    patType1correctionsJetResDown.extend([ cms.InputTag('pfMEtSysShiftCorr' + postfix) ])
                setattr(process, "patPFMetT1T2JetResDown" + postfix, getattr(process, "patPFMetT1T2JetResUp" + postfix).clone(
                    srcType1Corrections = cms.VInputTag(patType1correctionsJetResDown),
                    srcUnclEnergySums = cms.VInputTag(
                        cms.InputTag('patPFJetMETtype1p2CorrResDown' + postfix, 'type2' ),
                        cms.InputTag('patPFJetMETtype2CorrResDown' + postfix,   'type2' ),
                        cms.InputTag('patPFJetMETtype1p2CorrResDown' + postfix, 'offset'),
                        cms.InputTag('pfCandMETcorr' + postfix)
                    )
                ))
                metUncertaintySequence += getattr(process, "patPFMetT1T2JetResDown" + postfix)
                collectionsToKeep.append('patPFMetT1T2JetResDown' + postfix)

        #--------------------------------------------------------------------------------------------
        # shift "unclustered energy" (PFJets of Pt < 10 GeV plus PFCandidates not within jets)
        # and propagate effect of shift to (Type 1 as well as Type 1 + 2 corrected) MET
        #--------------------------------------------------------------------------------------------

        unclEnMETcorrections = [
            [ 'pfCandMETcorr' + postfix, [ '' ] ],
            [ 'patPFJetMETtype1p2Corr' + postfix, [ 'type2', 'offset' ] ],
            [ 'patPFJetMETtype2Corr' + postfix, [ 'type2' ] ],
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
            moduleUnclEnMETcorrUpName = "%sUnclusteredEnUp%s" % (baseName, postfix)
            setattr(process, moduleUnclEnMETcorrUpName, moduleUnclEnMETcorrUp)
            metUncertaintySequence += moduleUnclEnMETcorrUp
            unclEnMETcorrectionsUp.extend([ cms.InputTag(moduleUnclEnMETcorrUpName, instanceLabel)
                                            for instanceLabel in srcUnclEnMETcorr[1] ] )
            moduleUnclEnMETcorrDown = moduleUnclEnMETcorrUp.clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            )
            moduleUnclEnMETcorrDownName = "%sUnclusteredEnDown%s" % (baseName, postfix)
            setattr(process, moduleUnclEnMETcorrDownName, moduleUnclEnMETcorrDown)
            metUncertaintySequence += moduleUnclEnMETcorrDown
            unclEnMETcorrectionsDown.extend([ cms.InputTag(moduleUnclEnMETcorrDownName, instanceLabel)
                                              for instanceLabel in srcUnclEnMETcorr[1] ] )

        # propagate shifts in jet energy/resolution to "raw" (uncorrected) MET    
        setattr(process, "patPFMetUnclusteredEnUp" + postfix, getattr(process, metModNameT1 + postfix).clone(
            src = cms.InputTag('patPFMet' + postfix),
            srcType1Corrections = cms.VInputTag(unclEnMETcorrectionsUp)
        ))
        metUncertaintySequence += getattr(process, "patPFMetUnclusteredEnUp" + postfix)
        collectionsToKeep.append('patPFMetUnclusteredEnUp' + postfix)
        setattr(process, "patPFMetUnclusteredEnDown" + postfix, getattr(process, "patPFMetUnclusteredEnUp" + postfix).clone(
            srcType1Corrections = cms.VInputTag(unclEnMETcorrectionsDown)
        ))
        metUncertaintySequence += getattr(process, "patPFMetUnclusteredEnDown" + postfix)
        collectionsToKeep.append('patPFMetUnclusteredEnDown' + postfix)

        # propagate shifts in jet energy/resolution to Type 1 corrected MET
        setattr(process, "patPFMetT1UnclusteredEnUp" + postfix, getattr(process, metModNameT1 + postfix).clone(
            src = cms.InputTag(metModNameT1 + postfix),
            srcType1Corrections = cms.VInputTag(unclEnMETcorrectionsUp),
            srcUnclEnergySums = cms.VInputTag(),
            applyType2Corrections = cms.bool(False),
            type2CorrParameter = cms.PSet(
                A = cms.double(1.0)
            )
        ))
        metUncertaintySequence += getattr(process, "patPFMetT1UnclusteredEnUp" + postfix)
        collectionsToKeep.append('patPFMetT1UnclusteredEnUp' + postfix)
        setattr(process, "patPFMetT1UnclusteredEnDown" + postfix, getattr(process, "patPFMetT1UnclusteredEnUp" + postfix).clone(
            srcType1Corrections = cms.VInputTag(unclEnMETcorrectionsDown)
        ))
        metUncertaintySequence += getattr(process, "patPFMetT1UnclusteredEnDown" + postfix)
        collectionsToKeep.append('patPFMetT1UnclusteredEnDown' + postfix)
        
        # propagate shifts in jet energy/resolution to Type 1 + 2 corrected MET
        if makeType1p2corrPFMEt: 
            setattr(process, "patPFMetT1T2UnclusteredEnUp" + postfix, getattr(process, metModNameT1T2 + postfix).clone(
                srcUnclEnergySums = cms.VInputTag(
                    cms.InputTag('patPFJetMETtype1p2Corr' + postfix,                'type2' ),
                    cms.InputTag('patPFJetMETtype1p2CorrUnclusteredEnUp' + postfix, 'type2' ),
                    cms.InputTag('patPFJetMETtype2Corr' + postfix,                  'type2' ),   
                    cms.InputTag('patPFJetMETtype2CorrUnclusteredEnUp' + postfix,   'type2' ),
                    cms.InputTag('patPFJetMETtype1p2Corr' + postfix,                'offset'),
                    cms.InputTag('patPFJetMETtype1p2CorrUnclusteredEnUp' + postfix, 'offset'),
                    cms.InputTag('pfCandMETcorr' + postfix),
                    cms.InputTag('pfCandMETcorrUnclusteredEnUp' + postfix)
                )
            ))
            metUncertaintySequence += getattr(process, "patPFMetT1T2UnclusteredEnUp" + postfix)
            collectionsToKeep.append('patPFMetT1T2UnclusteredEnUp' + postfix)
            setattr(process, "patPFMetT1T2UnclusteredEnDown" + postfix, getattr(process, "patPFMetT1T2UnclusteredEnUp" + postfix).clone(
                srcUnclEnergySums = cms.VInputTag(
                    cms.InputTag('patPFJetMETtype1p2Corr' + postfix,                  'type2' ),
                    cms.InputTag('patPFJetMETtype1p2CorrUnclusteredEnDown' + postfix, 'type2' ),
                    cms.InputTag('patPFJetMETtype2Corr' + postfix,                    'type2' ),  
                    cms.InputTag('patPFJetMETtype2CorrUnclusteredEnDown' + postfix,   'type2' ),
                    cms.InputTag('patPFJetMETtype1p2Corr' + postfix,                  'offset'),
                    cms.InputTag('patPFJetMETtype1p2CorrUnclusteredEnDown' + postfix, 'offset'),
                    cms.InputTag('pfCandMETcorr' + postfix),
                    cms.InputTag('pfCandMETcorrUnclusteredEnDown' + postfix)
                )
            ))
            metUncertaintySequence += getattr(process, "patPFMetT1T2UnclusteredEnDown" + postfix)
            collectionsToKeep.append('patPFMetT1T2UnclusteredEnDown' + postfix)

        #--------------------------------------------------------------------------------------------    
        # propagate shifted electron/photon, muon and tau-jet energies to MET
        #--------------------------------------------------------------------------------------------

        metProducers = [ getattr(process, "patPFMet" + postfix),
                         getattr(process, metModNameT1 + postfix) ]
        if makeType1p2corrPFMEt:
            metProducers.append( getattr(process, metModNameT1T2 + postfix) )
        for metProducer in metProducers:
            
            if self._isValidInputTag(shiftedParticleCollections['electronCollection']):
                metCollectionsUp_Down = \
                    self._propagateMEtUncertainties(
                      process, shiftedParticleCollections['electronCollection'].value(), "Electron", "En",
                      shiftedParticleCollections['electronCollectionEnUp'], shiftedParticleCollections['electronCollectionEnDown'],
                      metProducer, "PF", metUncertaintySequence, postfix)
                collectionsToKeep.extend(metCollectionsUp_Down)

            if self._isValidInputTag(shiftedParticleCollections['photonCollection']):
                metCollectionsUp_Down = \
                    self._propagateMEtUncertainties(
                      process, shiftedParticleCollections['photonCollection'].value(), "Photon", "En",
                      shiftedParticleCollections['photonCollectionEnUp'], shiftedParticleCollections['photonCollectionEnDown'],
                      metProducer, "PF", metUncertaintySequence, postfix)
                collectionsToKeep.extend(metCollectionsUp_Down)
                
            if self._isValidInputTag(shiftedParticleCollections['muonCollection']):
                metCollectionsUp_Down = \
                    self._propagateMEtUncertainties(
                      process, shiftedParticleCollections['muonCollection'].value(), "Muon", "En",
                      shiftedParticleCollections['muonCollectionEnUp'], shiftedParticleCollections['muonCollectionEnDown'],
                      metProducer, "PF", metUncertaintySequence, postfix)
                collectionsToKeep.extend(metCollectionsUp_Down)

            if self._isValidInputTag(shiftedParticleCollections['tauCollection']):
                metCollectionsUp_Down = \
                    self._propagateMEtUncertainties(
                      process, shiftedParticleCollections['tauCollection'].value(), "Tau", "En",
                      shiftedParticleCollections['tauCollectionEnUp'], shiftedParticleCollections['tauCollectionEnDown'],
                      metProducer, "PF", metUncertaintySequence, postfix)
                collectionsToKeep.extend(metCollectionsUp_Down)

    def __call__(self, process,
                 electronCollection           = None,
                 photonCollection             = None,
                 muonCollection               = None,
                 tauCollection                = None,
                 jetCollection                = None,
                 dRjetCleaning                = None,
                 jetCorrLabel                 = None,
                 doSmearJets                  = None,                 
                 makeType1corrPFMEt           = None,
                 makeType1p2corrPFMEt         = None,
                 doApplyType0corr             = None,
                 sysShiftCorrParameter        = None,
                 doApplySysShiftCorr          = None,
                 jetSmearFileName             = None,
                 jetSmearHistogram            = None,
                 pfCandCollection             = None,
                 doApplyUnclEnergyCalibration = None,
                 jetCorrPayloadName           = None,
                 jetCorrLabelUpToL3           = None,
                 jetCorrLabelUpToL3Res        = None,
                 jecUncertaintyFile           = None,
                 jecUncertaintyTag            = None,
                 varyByNsigmas                = None,
                 addToPatDefaultSequence      = None,
                 outputModule                 = None,
                 postfix                      = None):
        JetMEtUncertaintyTools.__call__(
            self, process,
            electronCollection = electronCollection,
            photonCollection = photonCollection,
            muonCollection = muonCollection,
            tauCollection = tauCollection,
            jetCollection = jetCollection,
            jetCorrLabel = jetCorrLabel,
            doSmearJets = doSmearJets,
            jetSmearFileName = jetSmearFileName,
            jetSmearHistogram = jetSmearHistogram,
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
        if makeType1corrPFMEt is None:
            makeType1corrPFMEt = self._defaultParameters['makeType1corrPFMEt'].value
        if makeType1p2corrPFMEt is None:
            makeType1p2corrPFMEt = self._defaultParameters['makeType1p2corrPFMEt'].value
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
        pfCandCollection = self._initializeInputTag(pfCandCollection, 'pfCandCollection')
        if doApplyUnclEnergyCalibration is None:
            doApplyUnclEnergyCalibration = self._defaultParameters['doApplyUnclEnergyCalibration'].value

        self.setParameter('dRjetCleaning', dRjetCleaning)
        self.setParameter('makeType1corrPFMEt', makeType1corrPFMEt)
        self.setParameter('makeType1p2corrPFMEt', makeType1p2corrPFMEt)
        self.setParameter('doApplyType0corr', doApplyType0corr)
        self.setParameter('doApplySysShiftCorr', doApplySysShiftCorr)
        self.setParameter('sysShiftCorrParameter', sysShiftCorrParameter)
        self.setParameter('pfCandCollection', pfCandCollection)
        self.setParameter('doApplyUnclEnergyCalibration', doApplyUnclEnergyCalibration)
  
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
        doApplyType0corr = self._parameters['doApplyType0corr'].value
        sysShiftCorrParameter = self._parameters['sysShiftCorrParameter'].value
        doApplySysShiftCorr = self._parameters['doApplySysShiftCorr'].value        
        jetSmearFileName = self._parameters['jetSmearFileName'].value
        jetSmearHistogram = self._parameters['jetSmearHistogram'].value
        pfCandCollection = self._parameters['pfCandCollection'].value
        doApplyUnclEnergyCalibration = self._parameters['doApplyUnclEnergyCalibration'].value
        jetCorrPayloadName = self._parameters['jetCorrPayloadName'].value
        jetCorrLabelUpToL3 = self._parameters['jetCorrLabelUpToL3'].value
        jetCorrLabelUpToL3Res = self._parameters['jetCorrLabelUpToL3Res'].value
        jecUncertaintyFile = self._parameters['jecUncertaintyFile'].value
        jecUncertaintyTag = self._parameters['jecUncertaintyTag'].value
        varyByNsigmas = self._parameters['varyByNsigmas'].value
        addToPatDefaultSequence = self._parameters['addToPatDefaultSequence'].value
        outputModule = self._parameters['outputModule'].value
        postfix = self._parameters['postfix'].value

        if not hasattr(process, "pfType1MEtUncertaintySequence" + postfix):
            metUncertaintySequence = cms.Sequence()
            setattr(process, "pfType1MEtUncertaintySequence" + postfix, metUncertaintySequence)
        metUncertaintySequence = getattr(process, "pfType1MEtUncertaintySequence" + postfix)

        collectionsToKeep = []

        # produce collection of jets not overlapping with reconstructed
        # electrons/photons, muons and tau-jet candidates
        lastJetCollection, cleanedJetCollection = \
            self._addCleanedJets(process, jetCollection,
                                 electronCollection, photonCollection, muonCollection, tauCollection,
                                 metUncertaintySequence, postfix)
        
        # smear jet energies to account for difference in jet resolutions between MC and Data
        # (cf. JME-10-014 PAS)        
        jetCollectionResUp = None
        jetCollectionResDown = None
        if doSmearJets:
            lastJetCollection = \
              self._addSmearedJets(process, cleanedJetCollection, [ "smeared", jetCollection.value() ],
                                   jetSmearFileName, jetSmearHistogram, varyByNsigmas,
                                   uncertaintySequence = metUncertaintySequence, postfix = postfix)
                
            jetCollectionResUp = \
              self._addSmearedJets(process, cleanedJetCollection, [ "smeared", jetCollection.value(), "ResUp" ],
                                   jetSmearFileName, jetSmearHistogram, varyByNsigmas, -1., 
                                   uncertaintySequence = metUncertaintySequence, postfix = postfix)
            collectionsToKeep.append(jetCollectionResUp)
            jetCollectionResDown = \
              self._addSmearedJets(process, cleanedJetCollection, [ "smeared", jetCollection.value(), "ResDown" ],
                                   jetSmearFileName, jetSmearHistogram, varyByNsigmas, +1., 
                                   uncertaintySequence = metUncertaintySequence, postfix = postfix)
            collectionsToKeep.append(jetCollectionResDown)

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
                                              jetCollectionResUp, jetCollectionResDown,
                                              jetCorrLabelUpToL3, jetCorrLabelUpToL3Res,
                                              jecUncertaintyFile, jecUncertaintyTag,
                                              varyByNsigmas,
                                              postfix)
        setattr(process, "shiftedParticlesForType1PFMEtUncertainties" + postfix, shiftedParticleSequence)        
        metUncertaintySequence += getattr(process, "shiftedParticlesForType1PFMEtUncertainties" + postfix)
        collectionsToKeep.extend(addCollectionsToKeep)
        
        #--------------------------------------------------------------------------------------------    
        # propagate shifted particle energies to Type 1 and Type 1 + 2 corrected PFMET
        #--------------------------------------------------------------------------------------------

        self._addCorrPFMEt(process, metUncertaintySequence,
                           shiftedParticleCollections, pfCandCollection, doApplyUnclEnergyCalibration,
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
       
runType1PFMEtUncertainties = RunType1PFMEtUncertainties()
