import FWCore.ParameterSet.Config as cms

from FWCore.GuiBrowsers.ConfigToolBase import *

import PhysicsTools.PatAlgos.tools.helpers as configtools
from PhysicsTools.PatAlgos.tools.trigTools import _addEventContent
from PhysicsTools.PatUtils.tools.jmeUncertaintyTools import JetMEtUncertaintyTools
from PhysicsTools.PatUtils.tools.objectsUncertaintyTools import isValidInputTag,addSmearedJets

#from PhysicsTools.PatUtils.patPFMETCorrections_cff import *
import RecoMET.METProducers.METSigParams_cfi as jetResolutions
#from PhysicsTools.PatAlgos.producersLayer1.metProducer_cfi import patMETs

class RunNoPileUpMEtUncertainties(JetMEtUncertaintyTools):

    """ Shift energy of electrons, photons, muons, tau-jets and other jets
    reconstructed in the event up/down,
    in order to estimate effect of energy scale uncertainties on No-PU MET
    """
    _label = 'runNoPileUpMEtUncertainties'
    _defaultParameters = dicttypes.SortedKeysDict()
    def __init__(self):
        JetMEtUncertaintyTools.__init__(self)
        self.addParameter(self._defaultParameters, 'doApplyChargedHadronSubtraction', False,
                          "Flag to enable/disable usage of charged hadron subtraction when reconstructing jets", Type=bool)
	self.addParameter(self._defaultParameters, 'pfCandCollection', cms.InputTag('particleFlow'),
                          "Input PFCandidate collection", Type=cms.InputTag)
        self.addParameter(self._defaultParameters, 'doApplyUnclEnergyCalibration', False,
                          "Flag to enable/disable usage of 'unclustered energy' calibration", Type=bool)
        self.addParameter(self._defaultParameters, 'sfNoPUjetOffsetEnCorr', 0.0,
                          "Parameter of No-PU MET algorithm ", Type=float)
        self._parameters = copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def _addPFNoPUMEt(self, process, metUncertaintySequence,
                          shiftedParticleCollections, pfCandCollection, doApplyUnclEnergyCalibration,
                          sfNoPUjetOffsetEnCorr,
                          collectionsToKeep,
                          doApplyChargedHadronSubtraction,
                          doSmearJets,
                          jecUncertaintyFile, jecUncertaintyTag,
                          varyByNsigmas,
                          postfix):

        uncorrectedJetCollection = None
        smearedUncorrectedJetCollection = None
        correctedJetCollection = None
        smearedCorrectedJetCollection = None
        smearedPFCandidateCollection = None
        smearedPFCandToVertexAssociation = None
        puJetId = None
        puJetIdTag = None
        pfNoPUMEtData = None
        pfNoPUMEt = None
        pfNoPUMEtSequence = None
        patPFMetNoPileUp = None
        jetCorrLabelUpToL3 = None
        jetCorrLabelUpToL3Residual = None
        chsLabel = None

        if doApplyChargedHadronSubtraction:
            if not hasattr(process, "pfNoPUchsMEt"):
                process.load("RecoMET.METPUSubtraction.pfNoPUchsMEt_cff")
            uncorrectedJetCollection = 'ak4PFchsJets'
            smearedUncorrectedJetCollection = "smearedUncorrectedJetsForPFNoPUchsMEt"
            correctedJetCollection = 'calibratedAK4PFchsJetsForPFNoPUchsMEt'
            smearedCorrectedJetCollection = "smearedCorrectedJetsForPFNoPUchsMEt"
            smearedPFCandidateCollection = "smearedPFCandidatesForPFNoPUchsMEt"
            smearedPFCandToVertexAssociation = "smearedPFCandidateToVertexAssociationForPFNoPUchsMEt"
            puJetId = "puJetIdForPFNoPUchsMEt"
            puJetIdTag = "fullId"
            pfNoPUMEtData = "pfNoPUchsMEtData"
            pfNoPUMEt = "pfNoPUchsMEt"
            pfNoPUMEtSequence = "pfNoPUchsMEtSequence"
            patPFMetNoPileUp = "patPFNoPUchsMEt"
            jetCorrLabelUpToL3 = "ak4PFchsL1FastL2L3Corrector"
            jetCorrLabelUpToL3Residual = "ak4PFchsL1FastL2L3ResidualCorrector"
            chsLabel = "chs"
        else:
            if not hasattr(process, "pfNoPUMEt"):
                process.load("RecoMET.METPUSubtraction.pfNoPUMET_cff")
            uncorrectedJetCollection = 'ak4PFJets'
            smearedUncorrectedJetCollection = "smearedUncorrectedJetsForPFNoPUMEt"
            correctedJetCollection = 'calibratedAK4PFJetsForPFNoPUMEt'
            smearedCorrectedJetCollection = "smearedCorrectedJetsForPFNoPUMEt"
            smearedPFCandidateCollection = "smearedPFCandidatesForPFNoPUMEt"
            smearedPFCandToVertexAssociation = "smearedPFCandidateToVertexAssociationForPFNoPUMEt"
            puJetId = "puJetIdForPFNoPUMEt"
            puJetIdTag = "full53xId"
            pfNoPUMEtData = "pfNoPUMEtData"
            pfNoPUMEt = "pfNoPUMEt"
            pfNoPUMEtSequence = "pfNoPUMEtSequence"
            patPFMetNoPileUp = "patPFNoPUMEt"
            jetCorrLabelUpToL3 = "ak4PFL1FastL2L3Corrector"
            jetCorrLabelUpToL3Residual = "ak4PFL1FastL2L3ResidualCorrector"
            chsLabel = ""

        process.load("RecoMET.METPUSubtraction.pfNoPUMET_cff")
        if postfix != "":
            configtools.cloneProcessingSnippet(process, getattr(process, pfNoPUMEtSequence), postfix) #getattr(process, pfNoPUMEtSequence)

        # CV: set 'sfNoPUjetOffsetEnCorr' parameter to value
        #     that depends on whether data or MC is being processed
        pfNoPUMEtModule = getattr(process, pfNoPUMEt + postfix)
        pfNoPUMEtModule.sfNoPUjetOffsetEnCorr = cms.double(sfNoPUjetOffsetEnCorr)

        uncalibratedPFCandCollection = pfCandCollection.moduleLabel
        if doSmearJets:
            process.load("RecoJets.Configuration.GenJetParticles_cff")
            metUncertaintySequence += process.genParticlesForJetsNoNu
            process.load("RecoJets.Configuration.RecoGenJets_cff")
            metUncertaintySequence += process.ak4GenJetsNoNu
            setattr(process, smearedUncorrectedJetCollection + postfix, cms.EDProducer("SmearedPFJetProducer",
                src = cms.InputTag(uncorrectedJetCollection),
                jetCorrLabel = cms.InputTag(jetCorrLabelUpToL3.value()),
                dRmaxGenJetMatch = cms.string('min(0.5, 0.1 + 0.3*exp(-0.05*(genJetPt - 10.)))'),
                sigmaMaxGenJetMatch = cms.double(3.),
                inputFileName = cms.FileInPath('PhysicsTools/PatUtils/data/pfJetResolutionMCtoDataCorrLUT.root'),
                lutName = cms.string('pfJetResolutionMCtoDataCorrLUT'),
                jetResolutions = jetResolutions.METSignificance_params,
                skipRawJetPtThreshold = cms.double(10.), # GeV
                skipCorrJetPtThreshold = cms.double(1.e-2),
                srcGenJets = cms.InputTag('ak4GenJetsNoNu'),
                ##verbosity = cms.int32(1)
            ))
            metUncertaintySequence += getattr(process, smearedUncorrectedJetCollection + postfix)
            getattr(process, correctedJetCollection + postfix).src = cms.InputTag(smearedUncorrectedJetCollection + postfix)
            setattr(process, smearedPFCandidateCollection + postfix, cms.EDProducer("SmearedPFCandidateProducerForPFNoPUMEt",
                srcPFCandidates = pfCandCollection,
                srcJets = cms.InputTag(uncorrectedJetCollection),
                jetCorrLabel = cms.InputTag(jetCorrLabelUpToL3.value()),
                dRmaxGenJetMatch = cms.string('min(0.5, 0.1 + 0.3*exp(-0.05*(genJetPt - 10.)))'),
                sigmaMaxGenJetMatch = cms.double(3.),
                inputFileName = cms.FileInPath('PhysicsTools/PatUtils/data/pfJetResolutionMCtoDataCorrLUT.root'),
                lutName = cms.string('pfJetResolutionMCtoDataCorrLUT'),
                jetResolutions = jetResolutions.METSignificance_params,
                skipRawJetPtThreshold = cms.double(10.), # GeV
                skipCorrJetPtThreshold = cms.double(1.e-2),
                srcGenJets = cms.InputTag('ak4GenJetsNoNu'),
                ##verbosity = cms.int32(1)
            ))
            metUncertaintySequence += getattr(process, smearedPFCandidateCollection + postfix)
            uncalibratedPFCandCollection = smearedPFCandidateCollection + postfix
            setattr(process, smearedPFCandToVertexAssociation + postfix, getattr(process, "pfCandidateToVertexAssociationForPFNoPUMEt" + postfix).clone(
                PFCandidateCollection = cms.InputTag(smearedPFCandidateCollection + postfix)
            ))
            metUncertaintySequence += getattr(process, smearedPFCandToVertexAssociation + postfix)
            modulePFNoPUMEtData = getattr(process, pfNoPUMEtData + postfix) ##1
            setattr(modulePFNoPUMEtData, "srcPFCandidates", cms.InputTag(smearedPFCandidateCollection + postfix))
            setattr(modulePFNoPUMEtData, "srcPFCandToVertexAssociations", cms.InputTag(smearedPFCandToVertexAssociation + postfix))

        calibratedPFCandCollection = uncalibratedPFCandCollection
        calibratedPFCandToVertexAssociation = "pfCandidateToVertexAssociationForPFNoPUMEt" + postfix
        calibratedPFCandPFNoPUMEtData = pfNoPUMEtData + postfix
        if doApplyUnclEnergyCalibration:
            calibratedPFCandCollection = "calibratedPFCandidatesForPFNoPUMEt" + postfix
            setattr(process, calibratedPFCandCollection, cms.EDProducer("PFCandResidualCorrProducer",
                src = cms.InputTag(uncalibratedPFCandCollection),
                residualCorrLabel = cms.string(""),
                residualCorrEtaMax = cms.double(9.9),
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
            ))
            metUncertaintySequence += getattr(process, calibratedPFCandCollection)
            calibratedPFCandToVertexAssociation = "calibratedPFCandidateToVertexAssociationForPFNoPUMEt" + postfix
            setattr(process, calibratedPFCandToVertexAssociation, getattr(process, "pfCandidateToVertexAssociationForPFNoPUMEt" + postfix).clone(
                PFCandidateCollection = cms.InputTag(calibratedPFCandCollection)
            ))
            metUncertaintySequence += getattr(process, calibratedPFCandToVertexAssociation)
            calibratedPFCandPFNoPUMEtData = "calibratedPFCandPFNoPUMEtData" + postfix
            setattr(process, calibratedPFCandPFNoPUMEtData, getattr(process, "pfNoPUMEtData" + postfix).clone(
                srcPFCandidates = cms.InputTag(calibratedPFCandCollection),
                srcPFCandToVertexAssociations = cms.InputTag(calibratedPFCandToVertexAssociation),
            ))
            getattr(process, pfNoPUMEtSequence + postfix).replace(
              getattr(process, "pfNoPUMEtData" + postfix),
              getattr(process, "pfNoPUMEtData" + postfix) + getattr(process, calibratedPFCandPFNoPUMEtData))
            getattr(process, "pfNoPUMEt" + postfix).srcMVAMEtData = cms.InputTag(calibratedPFCandPFNoPUMEtData)
            getattr(process, "pfMETcorrType0ForPFNoPUMEt" + postfix).srcPFCandidateToVertexAssociations = cms.InputTag(calibratedPFCandToVertexAssociation)

        metUncertaintySequence += getattr(process, pfNoPUMEtSequence + postfix)

#        if doApplyChargedHadronSubtraction:
 #           self._addPATMEtProducer(process, metUncertaintySequence, 'pfNoPUchsMEt' + postfix, 'patPFchsMetNoPileUp', collectionsToKeep, postfix)
  #      else:
        self._addPATMEtProducer(process, metUncertaintySequence, 'pfNoPUMEt' + postfix, patPFMetNoPileUp, collectionsToKeep, postfix)



        variations = ['Up','Down']
        varDir= { 'Up':1., 'Down':-1. }

        #=====================================================
        # Leptons
        #=====================================================
        for leptonCollection in [ [ 'Electron', 'En', 'electronCollection', 0.3 ],
                                  [ 'Photon',   'En', 'photonCollection',   0.3 ],
                                  [ 'Muon',     'En', 'muonCollection',     0.3 ],
                                  [ 'Tau',      'En', 'tauCollection',      0.3 ] ]:
            if  ( leptonCollection[2] in shiftedParticleCollections ) and isValidInputTag(shiftedParticleCollections[leptonCollection[2]]):
                pfCandCollectionLeptonShift = \
                  self._addPFCandidatesForPFMEtInput(
                    process, metUncertaintySequence,
                    shiftedParticleCollections['%s' % leptonCollection[2]], leptonCollection[0], leptonCollection[1],
                    shiftedParticleCollections['%s%sUp' % (leptonCollection[2], leptonCollection[1])], shiftedParticleCollections['%s%sDown' % (leptonCollection[2], leptonCollection[1])],
                    leptonCollection[3],
                    cms.InputTag(calibratedPFCandCollection), "ForPFNoPU%sMEt%s" % (chsLabel, postfix))


                for var in variations:

                    modulePFCandidateToVertexAssociationShift = process.pfCandidateToVertexAssociation.clone(
                        PFCandidateCollection = cms.InputTag(pfCandCollectionLeptonShift[var])
                        )
                    modulePFCandidateToVertexAssociationShiftName = "pfCandidateToVertexAssociation%s%s%sForPileUpPF%sMEt%s" % (leptonCollection[0], leptonCollection[1], var, chsLabel, postfix)
                    setattr(process, modulePFCandidateToVertexAssociationShiftName, modulePFCandidateToVertexAssociationShift)
                    metUncertaintySequence += modulePFCandidateToVertexAssociationShift
                    uncorrectedJetsShiftName = "pfJets%s%s%sForPFNoPU%sMEt%s" % (leptonCollection[0], leptonCollection[1], var, chsLabel, postfix)
                    uncorrectedJetsShift = cms.EDProducer("ShiftedPFJetProducerByMatchedObject",
                                                      srcJets = cms.InputTag(uncorrectedJetCollection),
                                                      srcUnshiftedObjects = cms.InputTag(shiftedParticleCollections[ leptonCollection[2] ]),
                                                      srcShiftedObjects = cms.InputTag(shiftedParticleCollections['%s%sUp' % (leptonCollection[2], leptonCollection[1])]),
                                                      dRmatch_Jet = cms.double(leptonCollection[3])
                                                      )
                    setattr(process, uncorrectedJetsShiftName, uncorrectedJetsShift)
                    metUncertaintySequence += uncorrectedJetsShift
                    correctedJetsShift = "correctedJets%s%s%sForPFNoPU%sMEt%s" % (leptonCollection[0], leptonCollection[1], var, chsLabel, postfix)
                    setattr(process, correctedJetsShift, getattr(process, correctedJetCollection + postfix).clone(
                        src = cms.InputTag(uncorrectedJetsShiftName)
                        ))
                    metUncertaintySequence += getattr(process, correctedJetsShift)

                    puJetIdShift = "puJetId%s%s%sForPFNoPU%sMEt%s" % (leptonCollection[0], leptonCollection[1], var, chsLabel, postfix)
                    setattr(process, puJetIdShift, getattr(process, puJetId).clone(
                        jets = cms.InputTag(correctedJetsShift)
                        ))
                    metUncertaintySequence += getattr(process, puJetIdShift)
                    moduleMEtDataLeptonShift = getattr(process, calibratedPFCandPFNoPUMEtData).clone(
                        srcPFCandidates = cms.InputTag(pfCandCollectionLeptonShift[var]),
                        srcPFCandToVertexAssociations = cms.InputTag(modulePFCandidateToVertexAssociationShiftName),
                        srcJets = cms.InputTag(correctedJetsShift),
                        srcJetIds = cms.InputTag(puJetIdShift, puJetIdTag)
                        )
                    moduleMEtDataLeptonShiftName = "%s%s%s%s%s" % (pfNoPUMEtData, leptonCollection[0], leptonCollection[1], var, postfix)
                    setattr(process, moduleMEtDataLeptonShiftName, moduleMEtDataLeptonShift)
                    metUncertaintySequence += moduleMEtDataLeptonShift
                    moduleMEtLeptonShift = getattr(process, pfNoPUMEt + postfix).clone(
                        srcMVAMEtDataLeptonMatch = cms.InputTag(moduleMEtDataLeptonShiftName),
                        srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(
                            shiftedParticleCollections, leptonCollection[2], '%s%s%s' % (leptonCollection[2], leptonCollection[1], var), postfix = postfix))
                        )
                    moduleMEtLeptonShiftName = "%s%s%s%s%s" % (pfNoPUMEt, leptonCollection[0], leptonCollection[1], var, postfix)
                    setattr(process, moduleMEtLeptonShiftName, moduleMEtLeptonShift)
                    metUncertaintySequence += moduleMEtLeptonShift
                    self._addPATMEtProducer(process, metUncertaintySequence,
                                            moduleMEtLeptonShiftName, '%s%s%s%s' % (patPFMetNoPileUp, leptonCollection[0], leptonCollection[1], var),
                                            collectionsToKeep, postfix)


        if isValidInputTag(shiftedParticleCollections['jetCollection']):

            for var in variations:
                uncorrectedJetsEnShift = None
                correctedJetsEnShift = None
                if doApplyChargedHadronSubtraction:
                    uncorrectedJetsEnShift = "uncorrectedJetsEn%sForPFNoPUchsMEt%s" % (var, postfix)
                    correctedJetsEnShift = "correctedJetsEn%sForPFNoPUchsMEt%s" % (var, postfix)
                else:
                    uncorrectedJetsEnShift = "uncorrectedJetsEn%sForPFNoPUMEt%s" % (var, postfix)
                    correctedJetsEnShift = "correctedJetsEn%sForPFNoPUMEt%s" % (var, postfix)

                setattr(process, uncorrectedJetsEnShift, cms.EDProducer("ShiftedPFJetProducer",
                                                                        src = cms.InputTag(uncorrectedJetCollection),
                                                                        jetCorrInputFileName = cms.FileInPath(jecUncertaintyFile),
                                                                        jetCorrUncertaintyTag = cms.string(jecUncertaintyTag),
                                                                        addResidualJES = cms.bool(True),
                                                                        jetCorrLabelUpToL3 = cms.InputTag(jetCorrLabelUpToL3.value()),
                                                                        jetCorrLabelUpToL3Res = cms.InputTag(jetCorrLabelUpToL3Residual.value()),
                                                                        shiftBy = cms.double( varDir[var] *varyByNsigmas),
                                                                        ##verbosity = cms.int32(1)
                                                                        ))
                metUncertaintySequence += getattr(process, uncorrectedJetsEnShift)
                setattr(process, correctedJetsEnShift, getattr(process, uncorrectedJetsEnShift).clone(
                        src = cms.InputTag(correctedJetCollection + postfix),
                        addResidualJES = cms.bool(False)
                        ))
                metUncertaintySequence += getattr(process, correctedJetsEnShift)
                puJetIdJetEnShift = "%sJetEn%s%s" % (puJetId, var, postfix)
                setattr(process, puJetIdJetEnShift, getattr(process, puJetId + postfix).clone(
                        jets = cms.InputTag(correctedJetsEnShift)
                        ))
                metUncertaintySequence += getattr(process, puJetIdJetEnShift)
                pfNoPUMEtDataJetEnShift = "%sJetEn%s%s" % (pfNoPUMEtData, var, postfix)
                setattr(process, pfNoPUMEtDataJetEnShift, getattr(process, calibratedPFCandPFNoPUMEtData).clone(
                        srcJets = cms.InputTag(correctedJetsEnShift),
                        srcJetIds = cms.InputTag(puJetIdJetEnShift, puJetIdTag)
                        ))
                metUncertaintySequence += getattr(process, pfNoPUMEtDataJetEnShift)
                pfNoPUMEtJetEnShift = "%sJetEn%s%s" % (pfNoPUMEt, var, postfix)
                setattr(process, pfNoPUMEtJetEnShift, getattr(process, pfNoPUMEt + postfix).clone(
                        srcMVAMEtData = cms.InputTag(pfNoPUMEtDataJetEnShift),
                        srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections, postfix = postfix))
                        ))
                metUncertaintySequence += getattr(process, pfNoPUMEtJetEnShift)
                self._addPATMEtProducer(process, metUncertaintySequence,
                                        pfNoPUMEtJetEnShift, "%sJetEn%s" % (patPFMetNoPileUp, var), collectionsToKeep, postfix)


            if hasattr(process, smearedUncorrectedJetCollection + postfix):

                for var in variations:

                    smearedUncorrectedJetsResShift = None
                    calibratedJetsResShift = None
                    smearedPFCandidatesJetResShift = None
                    smearedPFCandToVertexAssociationJetResShift = None
                    if doApplyChargedHadronSubtraction:
                        smearedUncorrectedJetsResShift = "smearedUncorrectedJetsRes%sForPFNoPUchsMEt%s" % (var, postfix)
                        smearedPFCandidatesJetResShift = "smearedPFCandidatesJetRes%sForPFNoPUchsMEt%s" % (var, postfix)
                        smearedPFCandToVertexAssociationJetResShift = "smearedPFCandidateToVertexAssociationJetRes%sForPFNoPUchsMEt%s" % (var, postfix)
                    else:
                        smearedUncorrectedJetsResShift = "smearedUncorrectedJetsRes%sForPFNoPUMEt%s" % (var, postfix)
                        smearedPFCandidatesJetResShift = "smearedPFCandidatesJetRes%sForPFNoPUMEt%s" % (var, postfix)
                        smearedPFCandToVertexAssociationJetResShift = "smearedPFCandidateToVertexAssociationJetRes%sForPFNoPUMEt%s" % (var, postfix)
                    setattr(process, smearedUncorrectedJetsResShift, getattr(process, smearedUncorrectedJetCollection + postfix).clone(
                            shiftBy = cms.double(varDir[var]*varyByNsigmas)
                            ))
                    metUncertaintySequence += getattr(process, smearedUncorrectedJetsResShift)
                    correctedJetsResShift = correctedJetCollection.replace("Jets", "JetsRes"+var)
                    setattr(process, correctedJetsResShift + postfix, getattr(process, correctedJetCollection + postfix).clone(
                            src = cms.InputTag(smearedUncorrectedJetsResShift)
                            ))
                    metUncertaintySequence += getattr(process, correctedJetsResShift + postfix)
                    puJetIdJetResShift = "%sJetRes%s%s" % (puJetId, var, postfix)
                    setattr(process, puJetIdJetResShift, getattr(process, puJetId + postfix).clone(
                            jets = cms.InputTag(correctedJetsResShift + postfix)
                            ))
                    metUncertaintySequence += getattr(process, puJetIdJetResShift)
                    setattr(process, smearedPFCandidatesJetResShift, getattr(process, smearedPFCandidateCollection + postfix).clone(
                            shiftBy = cms.double(varDir[var]*varyByNsigmas)
                            ))
                    metUncertaintySequence += getattr(process, smearedPFCandidatesJetResShift)
                    setattr(process, smearedPFCandToVertexAssociationJetResShift, getattr(process, "pfCandidateToVertexAssociationForPFNoPUMEt" + postfix).clone(
                            PFCandidateCollection = cms.InputTag(smearedPFCandidatesJetResShift)
                            ))
                    metUncertaintySequence += getattr(process, smearedPFCandToVertexAssociationJetResShift)
                    pfNoPUMEtDataJetResShift = "%sJetRes%s%s" % (pfNoPUMEtData, var, postfix)
                    setattr(process, pfNoPUMEtDataJetResShift, getattr(process, calibratedPFCandPFNoPUMEtData).clone(
                            srcJets = cms.InputTag(correctedJetsResShift +postfix),
                            srcJetIds = cms.InputTag(puJetIdJetResShift, puJetIdTag),
                            srcPFCandidates = cms.InputTag(smearedPFCandidatesJetResShift),
                            srcPFCandToVertexAssociations = cms.InputTag(smearedPFCandToVertexAssociationJetResShift)
                            ))
                    metUncertaintySequence += getattr(process, pfNoPUMEtDataJetResShift)
                    pfNoPUMEtJetResShift = "%sJetRes%s%s" % (pfNoPUMEt, var, postfix)
                    setattr(process, pfNoPUMEtJetResShift, getattr(process, pfNoPUMEt + postfix).clone(
                            srcMVAMEtData = cms.InputTag(pfNoPUMEtDataJetResShift),
                            srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections, postfix = postfix))
                            ))
                    metUncertaintySequence += getattr(process, pfNoPUMEtJetResShift)
                    self._addPATMEtProducer(process, metUncertaintySequence,
                                            pfNoPUMEtJetResShift, "%sJetRes%s" % (patPFMetNoPileUp, var), collectionsToKeep, postfix)



                    pfCandsUnclusteredEnShift = None
                    pfCandidateToVertexAssociationUnclusteredEnShift = None
                    pfMETcorrType0UnclusteredEnShift = None
                    if doApplyChargedHadronSubtraction:
                        pfCandsUnclusteredEnShift = "pfCandsUnclusteredEn%sForPFNoPUchsMEt%s" % (var, postfix)
                        pfCandidateToVertexAssociationUnclusteredEnShift = "pfCandidateToVertexAssociationUnclusteredEn%sForPFNoPUchsMEt%s" % (var, postfix)
                        pfMETcorrType0UnclusteredEnShift = "pfMETcorrType0UnclusteredEn%sForPFNoPUchsMEt%s" % (var, postfix)
                    else:
                        pfCandsUnclusteredEnShift ="pfCandsUnclusteredEn%sForPFNoPUMEt%s" % (var, postfix)
                        pfCandidateToVertexAssociationUnclusteredEnShift = "pfCandidateToVertexAssociationUnclusteredEn%sForPFNoPUMEt%s" % (var, postfix)
                        pfMETcorrType0UnclusteredEnShift = "pfMETcorrType0UnclusteredEn%sForPFNoPUMEt%s" % (var, postfix)
                    setattr(process, pfCandsUnclusteredEnShift, cms.EDProducer("ShiftedPFCandidateProducerForPFNoPUMEt",
                                                                            srcPFCandidates = cms.InputTag(calibratedPFCandCollection),
                                                                            srcJets = cms.InputTag(correctedJetCollection + postfix),
                                                                            jetCorrInputFileName = cms.FileInPath(jecUncertaintyFile),
                                                                            jetCorrUncertaintyTag = cms.string(jecUncertaintyTag),
                                                                            minJetPt = cms.double(10.0),
                                                                            shiftBy = cms.double(varDir[var]*varyByNsigmas),
                                                                            unclEnUncertainty = cms.double(0.10)
                                                                            ))
                    metUncertaintySequence += getattr(process, pfCandsUnclusteredEnShift)
                    setattr(process, pfCandidateToVertexAssociationUnclusteredEnShift, process.pfCandidateToVertexAssociation.clone(
                            PFCandidateCollection = cms.InputTag(pfCandsUnclusteredEnShift)
                            ))
                    metUncertaintySequence += getattr(process, pfCandidateToVertexAssociationUnclusteredEnShift)
                    setattr(process, pfMETcorrType0UnclusteredEnShift, getattr(process, "pfMETcorrType0" + postfix).clone(
                            srcPFCandidateToVertexAssociations = cms.InputTag(pfCandidateToVertexAssociationUnclusteredEnShift)
                            ))
                    metUncertaintySequence += getattr(process, pfMETcorrType0UnclusteredEnShift)
                    pfNoPUMEtDataUnclusteredEnShift = "%sUnclusteredEn%s%s" % (pfNoPUMEtData, var, postfix)
                    setattr(process, pfNoPUMEtDataUnclusteredEnShift, getattr(process, calibratedPFCandPFNoPUMEtData).clone(
                            srcPFCandidates = cms.InputTag(pfCandsUnclusteredEnShift),
                            srcPFCandToVertexAssociations = cms.InputTag(pfCandidateToVertexAssociationUnclusteredEnShift)
                            ))
                    metUncertaintySequence += getattr(process, pfNoPUMEtDataUnclusteredEnShift)
                    pfNoPUMEtUnclusteredEnShift = "%sUnclusteredEn%s%s" % (pfNoPUMEt, var, postfix)
                    setattr(process, pfNoPUMEtUnclusteredEnShift, getattr(process, pfNoPUMEt + postfix).clone(
                            srcMVAMEtData = cms.InputTag(pfNoPUMEtDataUnclusteredEnShift),
                            srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections, postfix = postfix)),
                            srcType0Correction = cms.InputTag(pfMETcorrType0UnclusteredEnShift)
                            ))
                    metUncertaintySequence += getattr(process, pfNoPUMEtUnclusteredEnShift)
                    self._addPATMEtProducer(process, metUncertaintySequence,
                                            pfNoPUMEtUnclusteredEnShift, '%sUnclusteredEn%s' % (patPFMetNoPileUp, var), collectionsToKeep, postfix)

    def __call__(self, process,
                 electronCollection              = None,
                 photonCollection                = None,
                 muonCollection                  = None,
                 tauCollection                   = None,
                 jetCollection                   = None,
                 dRjetCleaning                   = None,
                 jetCorrLabel                    = None,
                 doApplyChargedHadronSubtraction = None,
                 doSmearJets                     = None,
                 jetSmearFileName                = None,
                 jetSmearHistogram               = None,
                 pfCandCollection                = None,
                 doApplyUnclEnergyCalibration    = None,
                 sfNoPUjetOffsetEnCorr           = None,
                 jetCorrPayloadName              = None,
                 jetCorrLabelUpToL3              = None,
                 jetCorrLabelUpToL3Res           = None,
                 jecUncertaintyFile              = None,
                 jecUncertaintyTag               = None,
                 varyByNsigmas                   = None,
                 addToPatDefaultSequence         = None,
                 outputModule                    = None,
                 postfix                         = None):
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
        if doApplyChargedHadronSubtraction is None:
            doApplyChargedHadronSubtraction = self._defaultParameters['doApplyChargedHadronSubtraction'].value
        pfCandCollection = self._initializeInputTag(pfCandCollection, 'pfCandCollection')
        if doApplyUnclEnergyCalibration is None:
            doApplyUnclEnergyCalibration = self._defaultParameters['doApplyUnclEnergyCalibration'].value
        if sfNoPUjetOffsetEnCorr is None:
            sfNoPUjetOffsetEnCorr = self._defaultParameters['sfNoPUjetOffsetEnCorr'].value

        self.setParameter('doApplyChargedHadronSubtraction', doApplyChargedHadronSubtraction)
        self.setParameter('pfCandCollection', pfCandCollection)
        self.setParameter('doApplyUnclEnergyCalibration', doApplyUnclEnergyCalibration)
        self.setParameter('sfNoPUjetOffsetEnCorr', sfNoPUjetOffsetEnCorr)

        self.apply(process)

    def toolCode(self, process):
        electronCollection = self._parameters['electronCollection'].value
        photonCollection = self._parameters['photonCollection'].value
        muonCollection = self._parameters['muonCollection'].value
        tauCollection = self._parameters['tauCollection'].value
        jetCollection = self._parameters['jetCollection'].value
        jetCorrLabel = self._parameters['jetCorrLabel'].value
        doApplyChargedHadronSubtraction = self._parameters['doApplyChargedHadronSubtraction'].value
        chsLabel = None
        if doApplyChargedHadronSubtraction:
            chsLabel = "chs"
        else:
            chsLabel = ""
        doSmearJets = self._parameters['doSmearJets'].value
        jetSmearFileName = self._parameters['jetSmearFileName'].value
        jetSmearHistogram = self._parameters['jetSmearHistogram'].value
        pfCandCollection = self._parameters['pfCandCollection'].value
        doApplyUnclEnergyCalibration = self._parameters['doApplyUnclEnergyCalibration'].value
        sfNoPUjetOffsetEnCorr = self._parameters['sfNoPUjetOffsetEnCorr'].value
        jetCorrPayloadName = self._parameters['jetCorrPayloadName'].value
        jetCorrLabelUpToL3 = self._parameters['jetCorrLabelUpToL3'].value
        jetCorrLabelUpToL3Res = self._parameters['jetCorrLabelUpToL3Res'].value
        jecUncertaintyFile = self._parameters['jecUncertaintyFile'].value
        jecUncertaintyTag = self._parameters['jecUncertaintyTag'].value
        varyByNsigmas = self._parameters['varyByNsigmas'].value
        addToPatDefaultSequence = self._parameters['addToPatDefaultSequence'].value
        outputModule = self._parameters['outputModule'].value
        postfix = self._parameters['postfix'].value

        if not hasattr(process, "pf%sNoPUMEtUncertaintySequence%s" % (chsLabel, postfix)):
            metUncertaintySequence = cms.Sequence()
            setattr(process, "pf%sNoPUMEtUncertaintySequence%s" % (chsLabel, postfix), metUncertaintySequence)
        metUncertaintySequence = getattr(process, "pf%sNoPUMEtUncertaintySequence%s" % (chsLabel, postfix))

        collectionsToKeep = []

        lastJetCollection = jetCollection.value()

        # smear jet energies to account for difference in jet resolutions between MC and Data
        # (cf. JME-10-014 PAS)
        if isValidInputTag(jetCollection):
            if doSmearJets:
                lastJetCollection = \
                    addSmearedJets(process, lastJetCollection, [ "smeared", jetCollection.value(), "ForPFNoPU%sMEt" % chsLabel ],
                                   jetSmearFileName, jetSmearHistogram, jetResolutions, varyByNsigmas, None,
                                   sequence = metUncertaintySequence, postfix = postfix)

            collectionsToKeep.append(lastJetCollection)

        #--------------------------------------------------------------------------------------------
        # produce collection of electrons/photons, muons, tau-jet candidates and jets
        # shifted up/down in energy by their respective energy uncertainties
        #--------------------------------------------------------------------------------------------

        shiftedParticleSequence, shiftedParticleCollections, addCollectionsToKeep = \
          self._addShiftedParticleCollections(process,
                                              electronCollection.value(),
                                              photonCollection.value(),
                                              muonCollection.value(),
                                              tauCollection.value(),
                                              jetCollection.value(), lastJetCollection, lastJetCollection,
                                              doSmearJets,
                                              jetCorrLabelUpToL3, jetCorrLabelUpToL3Res,
                                              jecUncertaintyFile, jecUncertaintyTag,
                                              jetSmearFileName, jetSmearHistogram,
                                              varyByNsigmas,
                                              "ForPFNoPU%sMEt%s" % (chsLabel, postfix))
        setattr(process, "shiftedParticlesForPFNoPU%sMEtUncertainties%s" % (chsLabel, postfix), shiftedParticleSequence)
        metUncertaintySequence += getattr(process, "shiftedParticlesForPFNoPU%sMEtUncertainties%s" % (chsLabel, postfix))
        collectionsToKeep.extend(addCollectionsToKeep)

        #--------------------------------------------------------------------------------------------
        # propagate shifted particle energies to No-PU MET
        #--------------------------------------------------------------------------------------------

        self._addPFNoPUMEt(process, metUncertaintySequence,
                               shiftedParticleCollections, pfCandCollection, doApplyUnclEnergyCalibration,
                               sfNoPUjetOffsetEnCorr,
                               collectionsToKeep,
                               doApplyChargedHadronSubtraction,
                               doSmearJets,
                               jecUncertaintyFile, jecUncertaintyTag,
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

runNoPileUpMEtUncertainties = RunNoPileUpMEtUncertainties()
