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
        self._parameters = copy.deepcopy(self._defaultParameters)
        self._comment = ""
   
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

        metUncertaintySequence += getattr(process, "producePatPFMETCorrections" + postfix)
        
        patType1correctionsCentralValue = [ cms.InputTag('patPFJetMETtype1p2Corr' + postfix, 'type1') ]
        if doApplyType0corr:
            patType1correctionsCentralValue.extend([ cms.InputTag('patPFMETtype0Corr' + postfix) ])
        if doApplySysShiftCorr:
            patType1correctionsCentralValue.extend([ cms.InputTag('pfMEtSysShiftCorr' + postfix) ])
        getattr(process, "patType1CorrectedPFMet" + postfix).srcType1Corrections = cms.VInputTag(patType1correctionsCentralValue)
        getattr(process, "patType1p2CorrectedPFMet" + postfix).srcType1Corrections = cms.VInputTag(patType1correctionsCentralValue)
        
        collectionsToKeep.extend([
            'patPFMet' + postfix,
            'patType1CorrectedPFMet' + postfix,
            'patType1p2CorrectedPFMet' + postfix])

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
            getattr(process, "producePatPFMETCorrections" + postfix).replace(getattr(process, "patPFMet" + postfix), smearedPatPFMetSequence)
            setattr(process, "patPFMet" + postfix, getattr(process, "patType1CorrectedPFMet" + postfix).clone(
                src = cms.InputTag('patPFMetForMEtUncertainty' + postfix),
                srcType1Corrections = cms.VInputTag(
                    cms.InputTag('patPFMETcorrJetSmearing' + postfix)
                )
            ))
            smearedPatPFMetSequence += getattr(process, "patPFMet" + postfix)
            metUncertaintySequence += smearedPatPFMetSequence 

        # propagate shifts in jet energy to "raw" (uncorrected) and Type 1 corrected MET
        metCollectionsUp_DownForRawMEt = \
            self._propagateMEtUncertainties(
              process, shiftedParticleCollections['lastJetCollection'], "Jet", "En",
              shiftedParticleCollections['jetCollectionEnUpForRawMEt'], shiftedParticleCollections['jetCollectionEnDownForRawMEt'],
              getattr(process, "patPFMet" + postfix), metUncertaintySequence, postfix)
        collectionsToKeep.extend(metCollectionsUp_DownForRawMEt)

        metCollectionsUp_DownForCorrMEt = \
            self._propagateMEtUncertainties(
              process, shiftedParticleCollections['lastJetCollection'], "Jet", "En",
              shiftedParticleCollections['jetCollectionEnUpForCorrMEt'], shiftedParticleCollections['jetCollectionEnDownForCorrMEt'],
              getattr(process, "patType1CorrectedPFMet" + postfix), metUncertaintySequence, postfix)
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
            setattr(process, "patType1p2CorrectedPFMetJetEnUp" + postfix, getattr(process, "patType1p2CorrectedPFMet" + postfix).clone(
                srcType1Corrections = cms.VInputTag(patType1correctionsJetEnUp),
                srcUnclEnergySums = cms.VInputTag(
                    cms.InputTag('patPFJetMETtype1p2CorrEnUp' + postfix, 'type2' ),
                    cms.InputTag('patPFJetMETtype2CorrEnUp' + postfix,   'type2' ),
                    cms.InputTag('patPFJetMETtype1p2CorrEnUp' + postfix, 'offset'),
                    cms.InputTag('pfCandMETcorr' + postfix)
                )
            ))
            metUncertaintySequence += getattr(process, "patType1p2CorrectedPFMetJetEnUp" + postfix)
            collectionsToKeep.append('patType1p2CorrectedPFMetJetEnUp' + postfix)
            patType1correctionsJetEnDown = [ cms.InputTag('patPFJetMETtype1p2CorrEnDown' + postfix, 'type1') ]
            if doApplyType0corr:
                patType1correctionsJetEnDown.extend([ cms.InputTag('patPFMETtype0Corr' + postfix) ])
            if doApplySysShiftCorr:
                patType1correctionsJetEnDown.extend([ cms.InputTag('pfMEtSysShiftCorr' + postfix) ])    
            setattr(process, "patType1p2CorrectedPFMetJetEnDown" + postfix, getattr(process, "patType1p2CorrectedPFMetJetEnUp" + postfix).clone(
                srcType1Corrections = cms.VInputTag(patType1correctionsJetEnDown),
                srcUnclEnergySums = cms.VInputTag(
                    cms.InputTag('patPFJetMETtype1p2CorrEnDown' + postfix, 'type2' ),
                    cms.InputTag('patPFJetMETtype2CorrEnDown' + postfix,   'type2' ),
                    cms.InputTag('patPFJetMETtype1p2CorrEnDown' + postfix, 'offset'),
                    cms.InputTag('pfCandMETcorr' + postfix)
                )
            ))
            metUncertaintySequence += getattr(process, "patType1p2CorrectedPFMetJetEnDown" + postfix)
            collectionsToKeep.append('patType1p2CorrectedPFMetJetEnDown' + postfix)

        if doSmearJets:
            # propagate shifts in jet resolution to "raw" (uncorrected) MET and Type 1 corrected MET
            for metProducer in [ getattr(process, "patPFMet" + postfix),
                                 getattr(process, "patType1CorrectedPFMet" + postfix) ]:

                metCollectionsUp_Down = \
                    self._propagateMEtUncertainties(
                      process, shiftedParticleCollections['lastJetCollection'], "Jet", "Res",
                      shiftedParticleCollections['jetCollectionResUp'], shiftedParticleCollections['jetCollectionResDown'],
                      metProducer, metUncertaintySequence, postfix)
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
                setattr(process, "patType1p2CorrectedPFMetJetResUp" + postfix,  getattr(process, "patType1p2CorrectedPFMet" + postfix).clone(
                    srcType1Corrections = cms.VInputTag(patType1correctionsJetResUp),
                    srcUnclEnergySums = cms.VInputTag(
                        cms.InputTag('patPFJetMETtype1p2CorrResUp' + postfix, 'type2' ),
                        cms.InputTag('patPFJetMETtype2CorrResUp' + postfix,   'type2' ),
                        cms.InputTag('patPFJetMETtype1p2CorrResUp' + postfix, 'offset'),
                        cms.InputTag('pfCandMETcorr' + postfix)
                    )
                ))
                metUncertaintySequence += getattr(process, "patType1p2CorrectedPFMetJetResUp" + postfix)
                collectionsToKeep.append('patType1p2CorrectedPFMetJetResUp' + postfix)
                patType1correctionsJetResDown = [ cms.InputTag('patPFJetMETtype1p2CorrResDown' + postfix, 'type1') ]
                if doApplyType0corr:
                    patType1correctionsJetResDown.extend([ cms.InputTag('patPFMETtype0Corr' + postfix) ])
                if doApplySysShiftCorr:
                    patType1correctionsJetResDown.extend([ cms.InputTag('pfMEtSysShiftCorr' + postfix) ])
                setattr(process, "patType1p2CorrectedPFMetJetResDown" + postfix, getattr(process, "patType1p2CorrectedPFMetJetResUp" + postfix).clone(
                    srcType1Corrections = cms.VInputTag(patType1correctionsJetResDown),
                    srcUnclEnergySums = cms.VInputTag(
                        cms.InputTag('patPFJetMETtype1p2CorrResDown' + postfix, 'type2' ),
                        cms.InputTag('patPFJetMETtype2CorrResDown' + postfix,   'type2' ),
                        cms.InputTag('patPFJetMETtype1p2CorrResDown' + postfix, 'offset'),
                        cms.InputTag('pfCandMETcorr' + postfix)
                    )
                ))
                metUncertaintySequence += getattr(process, "patType1p2CorrectedPFMetJetResDown" + postfix)
                collectionsToKeep.append('patType1p2CorrectedPFMetJetResDown' + postfix)

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
        setattr(process, "patPFMetUnclusteredEnUp" + postfix, getattr(process, "patType1CorrectedPFMet" + postfix).clone(
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
        setattr(process, "patType1CorrectedPFMetUnclusteredEnUp" + postfix, getattr(process, "patType1CorrectedPFMet" + postfix).clone(
            src = cms.InputTag('patType1CorrectedPFMet' + postfix),
            srcType1Corrections = cms.VInputTag(unclEnMETcorrectionsUp)
        ))
        metUncertaintySequence += getattr(process, "patType1CorrectedPFMetUnclusteredEnUp" + postfix)
        collectionsToKeep.append('patType1CorrectedPFMetUnclusteredEnUp' + postfix)
        setattr(process, "patType1CorrectedPFMetUnclusteredEnDown" + postfix, getattr(process, "patType1CorrectedPFMetUnclusteredEnUp" + postfix).clone(
            srcType1Corrections = cms.VInputTag(unclEnMETcorrectionsDown)
        ))
        metUncertaintySequence += getattr(process, "patType1CorrectedPFMetUnclusteredEnDown" + postfix)
        collectionsToKeep.append('patType1CorrectedPFMetUnclusteredEnDown' + postfix)
        
        # propagate shifts in jet energy/resolution to Type 1 + 2 corrected MET
        if makeType1p2corrPFMEt: 
            setattr(process, "patType1p2CorrectedPFMetUnclusteredEnUp" + postfix, getattr(process, "patType1p2CorrectedPFMet" + postfix).clone(
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
            metUncertaintySequence += getattr(process, "patType1p2CorrectedPFMetUnclusteredEnUp" + postfix)
            collectionsToKeep.append('patType1p2CorrectedPFMetUnclusteredEnUp' + postfix)
            setattr(process, "patType1p2CorrectedPFMetUnclusteredEnDown" + postfix, getattr(process, "patType1p2CorrectedPFMetUnclusteredEnUp" + postfix).clone(
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
            metUncertaintySequence += getattr(process, "patType1p2CorrectedPFMetUnclusteredEnDown" + postfix)
            collectionsToKeep.append('patType1p2CorrectedPFMetUnclusteredEnDown' + postfix)

        #--------------------------------------------------------------------------------------------    
        # propagate shifted electron/photon, muon and tau-jet energies to MET
        #--------------------------------------------------------------------------------------------

        metProducers = [ getattr(process, "patPFMet" + postfix),
                         getattr(process, "patType1CorrectedPFMet" + postfix) ]
        if makeType1p2corrPFMEt:
            metProducers.append( getattr(process, "patType1p2CorrectedPFMet" + postfix) )
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
                 doApplyType0corr        = None,
                 sysShiftCorrParameter   = None,
                 doApplySysShiftCorr     = None,
                 jetSmearFileName        = None,
                 jetSmearHistogram       = None,
                 pfCandCollection        = None,
                 jetCorrPayloadName      = None,
                 jetCorrLabelUpToL3      = None,
                 jetCorrLabelUpToL3Res   = None,
                 jecUncertaintyFile      = None,
                 jecUncertaintyTag       = None,
                 varyByNsigmas           = None,
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

        self.setParameter('dRjetCleaning', dRjetCleaning)
        self.setParameter('makeType1corrPFMEt', makeType1corrPFMEt)
        self.setParameter('makeType1p2corrPFMEt', makeType1p2corrPFMEt)
        self.setParameter('doApplyType0corr', doApplyType0corr)
        self.setParameter('doApplySysShiftCorr', doApplySysShiftCorr)
        self.setParameter('sysShiftCorrParameter', sysShiftCorrParameter)
        self.setParameter('pfCandCollection', pfCandCollection)
  
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
