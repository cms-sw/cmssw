import FWCore.ParameterSet.Config as cms

from FWCore.GuiBrowsers.ConfigToolBase import *

from PhysicsTools.PatAlgos.producersLayer1.metProducer_cfi import patMETs
from PhysicsTools.PatAlgos.tools.trigTools import _addEventContent
from PhysicsTools.PatUtils.tools.jmeUncertaintyTools import JetMEtUncertaintyTools, addSmearedJets, isValidInputTag
from PhysicsTools.PatUtils.tools.propagateMEtUncertainties import *

import RecoMET.METProducers.METSigParams_cfi as jetResolutions



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
        self.addParameter(self._defaultParameters, 'makeType1p2corrPFMEt', False,
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
        self.addParameter(self._defaultParameters, 'jetCollectionUnskimmed', None,
                          "Unskimmed jets for type1 and type2 computations", Type=cms.InputTag, acceptNoneValue=True)
        self._parameters = copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def _addCorrPFMEt(self, process, metUncertaintySequence,
                      shiftedParticleCollections, pfCandCollection,jetCollectionUnskimmed,
                      doApplyUnclEnergyCalibration,
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


        # loading default files
        if not hasattr(process, 'producePatPFMETCorrectionsUnc'):
            process.load("PhysicsTools.PatUtils.patPFMETCorrections_cff")

       
        #
        #protections against inconsistent met correction scheme :
        #
        if makeType1p2corrPFMEt and not makeType1corrPFMEt :
            print "WARNING: Type2 correction called without Type1 => Type1 enabled automatically for consistency"
            makeType1corrPFMEt = True
        if doApplyType0corr and not makeType1corrPFMEt :
            print "WARNING: Type0 correction called without Type1 => Type0 disabled automatically for consistency"
            doApplyType0corr = False

        #
        # Assign MET names, and create the modules and the sequence used
        #
        metModName, metModNameT1, metModNameT1T2, collectionsToKeep = \
            createPatMETModules(process, "PF", getattr(process, "producePatPFMETCorrectionsUnc"),
                                makeType1corrPFMEt, makeType1p2corrPFMEt, doApplyType0corr,
                                doApplySysShiftCorr, doApplyUnclEnergyCalibration,
                                sysShiftCorrParameter, "")



        # If with empty postfix, make a backup of
        # the met sequence used, because the original
        # sequence will be modified later in this function
        if postfix == "":
            configtools.cloneProcessingSnippet(process, process.producePatPFMETCorrectionsUnc , "OriginalReserved")
        else:
            if postfix == "OriginalReserved":
                raise ValueError("Postfix label '%s' is reserved for internal usage !!" % postfix)

            if hasattr(process, "producePatPFMETCorrectionsUncOriginalReserved"):
                configtools.cloneProcessingSnippet(process, process.producePatPFMETCorrectionsUnc, postfix, removePostfix="OriginalReserved")
            else:
                configtools.cloneProcessingSnippet(process, process.producePatPFMETCorrectionsUnc, postfix)

        metUncertaintySequence += getattr(process, "producePatPFMETCorrectionsUnc" + postfix)

        #
        # prepare smeared jets variations if needed
        #
        if doSmearJets:
            self._prepareJetVariationsForMET(process, "type1p2","Res",shiftedParticleCollections,metUncertaintySequence, postfix)
            if makeType1p2corrPFMEt:
                self._prepareJetVariationsForMET(process, "type2","Res",shiftedParticleCollections,metUncertaintySequence, postfix)

        #
        # prepare energy variations
        #
        self._prepareJetVariationsForMET(process, "type1p2","En",shiftedParticleCollections,metUncertaintySequence, postfix)
        if makeType1p2corrPFMEt:
            self._prepareJetVariationsForMET(process, "type2","En",shiftedParticleCollections,metUncertaintySequence, postfix)


        #
        # apply MET smearing to "raw" (uncorrected) MET
        #
        if doSmearJets:
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


        #--------------------------------------------------------------------------------------------
        # propagate shifts in jet energy scale
        #--------------------------------------------------------------------------------------------

        # to "raw" (uncorrected) and Type corrected MET
        metTypes = {} #jet identifiers
        metTypes[ metModName ]='ForRawMEt'
        if makeType1corrPFMEt:
            metTypes[ metModNameT1 ]=''



        for met in metTypes.keys():
            metCollectionsUp_Down = \
                propagateMEtUncertainties(
                process, shiftedParticleCollections['lastJetCollection'], "Jet", "En",
                shiftedParticleCollections['jetCollectionEnUp' + metTypes[met] ], shiftedParticleCollections['jetCollectionEnDown' + metTypes[met] ],
                getattr(process, met + postfix), "PF", metUncertaintySequence, postfix)
            collectionsToKeep.extend(metCollectionsUp_Down)

        #  to Type 1 + 2 corrected MET
        if makeType1p2corrPFMEt:
            self._propagateJetVariationsToT1T2Met(process, "En", metUncertaintySequence, jetCorrLabel,
                                                  metModNameT1T2, doApplyType0corr, doApplySysShiftCorr,
                                                  postfix)


        #--------------------------------------------------------------------------------------------
        # propagate shifts in jet energy resolution
        #--------------------------------------------------------------------------------------------
        if doSmearJets:

            #  to "raw" (uncorrected) MET and to Type 1 corrected MET if asked
            metProducers = [ getattr(process, metModName + postfix) ]
            if makeType1corrPFMEt:
                metProducers.append( getattr(process, metModNameT1 + postfix) )

            jetERShiftedMETs = propagateERShiftedJets(process, shiftedParticleCollections,
                                                      metProducers, "PF", metUncertaintySequence, postfix)
            collectionsToKeep.extend( jetERShiftedMETs );


            # to Type 1 + 2 corrected MET
            if makeType1p2corrPFMEt:
                self._propagateJetVariationsToT1T2Met(process, "Res", metUncertaintySequence,jetCorrLabel,
                                                      metModNameT1T2, doApplyType0corr, doApplySysShiftCorr,
                                                      postfix)

        #--------------------------------------------------------------------------------------------
        # shift "unclustered energy" (PFJets of Pt < 10 GeV plus PFCandidates not within jets)

        #--------------------------------------------------------------------------------------------
        #unclEnMETcorrections = {}
        unclEnMETcorrections = \
            self._prepareShiftedUnclusteredEnergy(process, metUncertaintySequence,
                                                  varyByNsigmas, postfix)



        #--------------------------------------------------------------------------------------------
        # and propagate effect of shift to (Type 1 as well as Type 1 + 2 corrected) MET
        #--------------------------------------------------------------------------------------------

        variations=["Up","Down"]
        for var in variations:

            # propagate shifts in jet energy/resolution to "raw" (uncorrected) MET
            setattr(process, "patPFMetUnclusteredEn" + var + postfix, getattr(process, metModNameT1 + postfix).clone(
                    src = cms.InputTag('patPFMet' + postfix),
                    srcType1Corrections = cms.VInputTag(unclEnMETcorrections[ var ])
                    ))
            metUncertaintySequence += getattr(process, "patPFMetUnclusteredEn" + var + postfix)
            collectionsToKeep.append('patPFMetUnclusteredEn' + var + postfix)

            # propagate shifts in jet energy/resolution to Type 1 corrected MET
            if makeType1corrPFMEt:
                setattr(process, metModNameT1+"UnclusteredEn" + var + postfix, getattr(process, metModNameT1 + postfix).clone(
                        src = cms.InputTag(metModNameT1 + postfix),
                        srcType1Corrections = cms.VInputTag(unclEnMETcorrections[ var ]),
                        srcUnclEnergySums = cms.VInputTag(),
                        applyType2Corrections = cms.bool(False),
                        type2CorrParameter = cms.PSet(
                            A = cms.double(1.0)
                            )
                        ))
                metUncertaintySequence += getattr(process, metModNameT1+"UnclusteredEn" + var + postfix)
                collectionsToKeep.append(metModNameT1+'UnclusteredEn' + var + postfix)

            # propagate shifts in jet energy/resolution to Type 1 + 2 corrected MET
            if makeType1p2corrPFMEt:
                setattr(process, metModNameT1T2+"UnclusteredEn" + var + postfix, getattr(process, metModNameT1T2 + postfix).clone(
                        srcUnclEnergySums = cms.VInputTag(
                            cms.InputTag('patPFJetMETtype1p2Corr' + postfix,                'type2' ),
                            cms.InputTag('patPFJetMETtype1p2CorrUnclusteredEn' + var + postfix, 'type2' ),
                            cms.InputTag('patPFJetMETtype2Corr' + postfix,                  'type2' ),
                            cms.InputTag('patPFJetMETtype2CorrUnclusteredEn' + var + postfix,   'type2' ),
                            cms.InputTag('patPFJetMETtype1p2Corr' + postfix,                'offset'),
                            cms.InputTag('patPFJetMETtype1p2CorrUnclusteredEn' + var + postfix, 'offset'),
                            cms.InputTag('pfCandMETcorr' + postfix),
                            cms.InputTag('pfCandMETcorrUnclusteredEn' + var + postfix)
                            )
                        ))
                metUncertaintySequence += getattr(process, metModNameT1T2+"UnclusteredEn" + var + postfix)
                collectionsToKeep.append(metModNameT1T2+'UnclusteredEn' + var + postfix)


        #--------------------------------------------------------------------------------------------
        # propagate shifted electron/photon, muon and tau-jet energies to MET
        #--------------------------------------------------------------------------------------------

        metProducers = [ getattr(process, "patPFMet" + postfix) ]
        if makeType1corrPFMEt:
            metProducers.append( getattr(process, metModNameT1 + postfix) )
        if makeType1p2corrPFMEt:
            metProducers.append( getattr(process, metModNameT1T2 + postfix) )


        singleParticleShiftedMETs = propagateShiftedSingleParticles(process, shiftedParticleCollections, metProducers, "PF", metUncertaintySequence, postfix)
        collectionsToKeep.extend( singleParticleShiftedMETs );


        #fix the default jets for the type1 computation to those used to compute the uncertainties
        #in order to be consistent with what is done in the correction and uncertainty step
        #particularly true for miniAODs
        if isValidInputTag(jetCollectionUnskimmed):
            getattr(process,"patPFJetMETtype1p2Corr").src = jetCollectionUnskimmed
            getattr(process,"patPFJetMETtype2Corr").src = jetCollectionUnskimmed



    def _prepareShiftedUnclusteredEnergy(self, process, metUncertaintySequence, varyByNsigmas, postfix):

        # define all corrections taht will need a shifting of the unclustered energy
        unclEnMETcorrectionsSrcs = [
            [ 'pfCandMETcorr' + postfix, [ '' ] ],
            [ 'patPFJetMETtype1p2Corr' + postfix, [ 'type2', 'offset' ] ],
            [ 'patPFJetMETtype2Corr' + postfix, [ 'type2' ] ],
            ]

        unclEnMETcorrections = {}

        #
        #create each module to perform the upper variations of each type of correction
        #
        variations=["Up","Down"]
        for var in variations:
            shiftDir=1.
            if var=="Down":
                shiftDir=-1.

            for srcUnclEnMETcorr in unclEnMETcorrectionsSrcs:
                moduleUnclEnMETcorr = cms.EDProducer("ShiftedMETcorrInputProducer",
                                                       src = cms.VInputTag(
                        [ cms.InputTag(srcUnclEnMETcorr[0], instanceLabel) for instanceLabel in srcUnclEnMETcorr[1] ]
                        ),
                                                       uncertainty = cms.double(0.10),
                                                       shiftBy = cms.double(shiftDir*varyByNsigmas)
                                                       )
                baseName = srcUnclEnMETcorr[0]
                if postfix != "":
                    if baseName[-len(postfix):] == postfix:
                        baseName = baseName[0:-len(postfix)]
                    else:
                        raise StandardError("Tried to remove postfix %s from label %s, but it wasn't there" % (postfix, baseName))
                moduleUnclEnMETcorrName = "%sUnclusteredEn%s%s" % (baseName, var, postfix)
                setattr(process, moduleUnclEnMETcorrName, moduleUnclEnMETcorr)
                metUncertaintySequence += moduleUnclEnMETcorr
                unclEnMETcorrections[ var ] = ([ cms.InputTag(moduleUnclEnMETcorrName, instanceLabel)
                                                 for instanceLabel in srcUnclEnMETcorr[1] ] )

        return unclEnMETcorrections


    def _prepareJetVariationsForMET(self,process, identifier, varType,shiftedParticleCollections, metUncertaintySequence, postfix):

        variations=["Up","Down"]
        for var in variations:
            setattr(process, "selectedPatJetsForMET"+identifier+"Corr"+ varType + var + postfix,
                    getattr(process, shiftedParticleCollections['jetCollection' + varType + var]).clone(
                    src = cms.InputTag('selectedPatJetsForMET'+identifier+'Corr' + postfix)
                    ))
            metUncertaintySequence += getattr(process, "selectedPatJetsForMET"+identifier+"Corr"+ varType + var + postfix)


    def _propagateJetVariationsToT1T2Met(self,process, varType, metUncertaintySequence,jetCorrLabel, metModNameT1T2, doApplyType0corr, doApplySysShiftCorr, postfix):

        collectionsToKeep = []

        variations=["Up","Down"]
        for var in variations:
            setattr(process, "patPFJetMETtype1p2Corr" + varType + var + postfix, getattr(process, "patPFJetMETtype1p2Corr" + postfix).clone(
                src = cms.InputTag(getattr(process, "selectedPatJetsForMETtype1p2Corr" + varType + var + postfix).label()),
                jetCorrLabel = cms.InputTag(jetCorrLabel.value())
            ))
            metUncertaintySequence += getattr(process, "patPFJetMETtype1p2Corr" + varType + var + postfix)
            setattr(process, "patPFJetMETtype2Corr" + varType + var + postfix, getattr(process, "patPFJetMETtype2Corr" + postfix).clone(
                src = cms.InputTag('selectedPatJetsForMETtype2Corr' + varType + var + postfix)
            ))
            metUncertaintySequence += getattr(process, "patPFJetMETtype2Corr" + varType + var + postfix)

            patType1correctionsJetEnUp = [ cms.InputTag('patPFJetMETtype1p2Corr'+ varType + var + postfix, 'type1') ]
            if doApplyType0corr:
                patType1correctionsJetEnUp.extend([ cms.InputTag('patPFMETtype0Corr' + postfix) ])
            if doApplySysShiftCorr:
                patType1correctionsJetEnUp.extend([ cms.InputTag('pfMEtSysShiftCorr' + postfix) ])
            setattr(process, metModNameT1T2+"Jet" + varType + var + postfix, getattr(process, metModNameT1T2 + postfix).clone(
                    srcType1Corrections = cms.VInputTag(patType1correctionsJetEnUp),
                    srcUnclEnergySums = cms.VInputTag(
                        cms.InputTag('patPFJetMETtype1p2Corr' + varType + var + postfix, 'type2' ),
                        cms.InputTag('patPFJetMETtype2Corr' + varType + var + postfix,   'type2' ),
                        cms.InputTag('patPFJetMETtype1p2Corr' + varType + var + postfix, 'offset'),
                        cms.InputTag('pfCandMETcorr' + postfix)
                        ),
                    applyType2Corrections = cms.bool(True),
                    type2CorrParameter = cms.PSet(
                        A = cms.double(1.4)
                        )
                    ))
            metUncertaintySequence += getattr(process, metModNameT1T2+"Jet" + varType + var + postfix)
            collectionsToKeep.append(metModNameT1T2+'Jet' + varType + var  + postfix)



    def __call__(self, process,
                 electronCollection           = None,
                 photonCollection             = None,
                 muonCollection               = None,
                 tauCollection                = None,
                 jetCollection                = None,
                 jetCollectionUnskimmed       = None,
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
           # jetCollectionUnskimmed = jetCollectionUnskimmed,
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
       
        jetCollectionUnskimmed  = self._initializeInputTag(jetCollectionUnskimmed, 'jetCollectionUnskimmed')

        self.setParameter('dRjetCleaning', dRjetCleaning)
        self.setParameter('makeType1corrPFMEt', makeType1corrPFMEt)
        self.setParameter('makeType1p2corrPFMEt', makeType1p2corrPFMEt)
        self.setParameter('doApplyType0corr', doApplyType0corr)
        self.setParameter('doApplySysShiftCorr', doApplySysShiftCorr)
        self.setParameter('sysShiftCorrParameter', sysShiftCorrParameter)
        self.setParameter('pfCandCollection', pfCandCollection)
        self.setParameter('doApplyUnclEnergyCalibration', doApplyUnclEnergyCalibration)
        self.setParameter('jetCollectionUnskimmed', jetCollectionUnskimmed)

        #specific postfix for miniAODs
        #temporary fix for 74X on the handling of JER uncertainties
        if doSmearJets:
            postfix="Smeared"

        self.apply(process)

    def toolCode(self, process):
        electronCollection = self._parameters['electronCollection'].value
        photonCollection = self._parameters['photonCollection'].value
        muonCollection = self._parameters['muonCollection'].value
        tauCollection = self._parameters['tauCollection'].value
        jetCollection = self._parameters['jetCollection'].value
        jetCollectionUnskimmed = self._parameters['jetCollectionUnskimmed'].value
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


        if isValidInputTag(jetCollection):
            # produce collection of jets not overlapping with reconstructed
            # electrons/photons, muons and tau-jet candidates
            lastJetCollection, cleanedJetCollection = \
                self._addCleanedJets(process, jetCollection,
                                     electronCollection, photonCollection, muonCollection, tauCollection,
                                     metUncertaintySequence, postfix)

            # smear jet energies to account for difference in jet resolutions between MC and Data
            # (cf. JME-10-014 PAS)
            if doSmearJets:
                lastJetCollection = \
                    addSmearedJets(process, cleanedJetCollection,
                                   [ "smeared", jetCollection.value() ],
                                   jetSmearFileName, jetSmearHistogram, jetResolutions,
                                   varyByNsigmas, None, metUncertaintySequence, postfix)
                collectionsToKeep.append( lastJetCollection )

            #for default w/o smearing
            collectionsToKeep.append( lastJetCollection )

        #--------------------------------------------------------------------------------------------
        # produce collection of electrons/photons, muons, tau-jet candidates and jets
        # shifted up/down in energy by their respective energy uncertainties
        #--------------------------------------------------------------------------------------------
        shiftedParticleSequence, shiftedParticleCollections, addCollectionsToKeep = \
          self._addShiftedParticleCollections(process,
                                              electronCollection.value() if isValidInputTag(electronCollection) else "",
                                              photonCollection.value() if isValidInputTag(photonCollection) else "",
                                              muonCollection.value() if isValidInputTag(muonCollection) else "",
                                              tauCollection.value() if isValidInputTag(tauCollection) else "",
                                              jetCollection.value() if isValidInputTag(jetCollection) else "",
                                              cleanedJetCollection,
                                              lastJetCollection, doSmearJets,
                                              jetCorrLabelUpToL3, jetCorrLabelUpToL3Res,
                                              jecUncertaintyFile, jecUncertaintyTag,
                                              jetSmearFileName, jetSmearHistogram,
                                              varyByNsigmas, postfix)
        setattr(process, "shiftedParticlesForType1PFMEtUncertainties" + postfix, shiftedParticleSequence)
        metUncertaintySequence += getattr(process, "shiftedParticlesForType1PFMEtUncertainties" + postfix)
        collectionsToKeep.extend(addCollectionsToKeep)



        #--------------------------------------------------------------------------------------------
        # propagate shifted particle energies to Type 1 and Type 1 + 2 corrected PFMET
        #--------------------------------------------------------------------------------------------

        self._addCorrPFMEt(process, metUncertaintySequence,
                           shiftedParticleCollections, pfCandCollection, jetCollectionUnskimmed,
                           doApplyUnclEnergyCalibration,
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
