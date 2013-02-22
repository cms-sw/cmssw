import FWCore.ParameterSet.Config as cms

from FWCore.GuiBrowsers.ConfigToolBase import *

import PhysicsTools.PatAlgos.tools.helpers as configtools
from PhysicsTools.PatAlgos.tools.trigTools import _addEventContent
from PhysicsTools.PatUtils.tools.jmeUncertaintyTools import JetMEtUncertaintyTools

from PhysicsTools.PatUtils.patPFMETCorrections_cff import *
import RecoMET.METProducers.METSigParams_cfi as jetResolutions
from PhysicsTools.PatAlgos.producersLayer1.metProducer_cfi import patMETs
 
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
                          "Flag to enable/disable usage of charged hadron subtraction when reconstructing jets", Type = bool)
	self.addParameter(self._defaultParameters, 'pfCandCollection', cms.InputTag('particleFlow'), 
                          "Input PFCandidate collection", Type = cms.InputTag)	
        self._parameters = copy.deepcopy(self._defaultParameters)
        self._comment = ""

    def _addNoPileUpPFMEt(self, process, metUncertaintySequence,
                          shiftedParticleCollections, pfCandCollection,
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
        puJetId = None
        noPileUpPFMEtData = None
        noPileUpPFMEt = None
        noPileUpPFMEtSequence = None
        patPFMetNoPileUp = None
        jetCorrLabelUpToL3 = None
        jetCorrLabelUpToL3Residual = None
        chsLabel = None
        if doApplyChargedHadronSubtraction:
            if not hasattr(process, "noPileUpPFchsMEt"):
                process.load("JetMETCorrections.METPUSubtraction.noPileUpPFchsMEt_cff")            
            uncorrectedJetCollection = 'ak5PFchsJets'
            smearedUncorrectedJetCollection = "smearedUncorrectedJetsForNoPileUpPFchsMEt"
            correctedJetCollection = 'calibratedAK5PFchsJetsForNoPileUpPFchsMEt'
            smearedCorrectedJetCollection = "smearedCorrectedJetsForNoPileUpPFchsMEt"
            puJetId = "puJetIdForNoPileUpPFchsMEt"
            noPileUpPFMEtData = "noPileUpPFchsMEtData"
            noPileUpPFMEt = "noPileUpPFchsMEt"
            noPileUpPFMEtSequence = "noPileUpPFchsMEtSequence"
            patPFMetNoPileUp = "patPFchsMetNoPileUp"
            jetCorrLabelUpToL3 = "ak5PFchsL1FastL2L3"
            jetCorrLabelUpToL3Residual = "ak5PFchsL1FastL2L3Residual"
            chsLabel = "chs"
        else:
            if not hasattr(process, "noPileUpPFMEt"):
                process.load("JetMETCorrections.METPUSubtraction.noPileUpPFMET_cff")            
            uncorrectedJetCollection = 'ak5PFJets'
            smearedUncorrectedJetCollection = "smearedUncorrectedJetsForNoPileUpPFMEt" 
            correctedJetCollection = 'calibratedAK5PFJetsForNoPileUpPFMEt'
            smearedCorrectedJetCollection = "smearedCorrectedJetsForNoPileUpPFMEt"
            puJetId = "puJetIdForNoPileUpPFMEt"
            noPileUpPFMEtData = "noPileUpPFMEtData"
            noPileUpPFMEt = "noPileUpPFMEt"
            noPileUpPFMEtSequence = "noPileUpPFMEtSequence"
            patPFMetNoPileUp = "patPFMetNoPileUp"
            jetCorrLabelUpToL3 = "ak5PFL1FastL2L3"
            jetCorrLabelUpToL3Residual = "ak5PFL1FastL2L3Residual"
            chsLabel = ""

        if postfix != "":
            configtools.cloneProcessingSnippet(process, getattr(process, "noPileUpPFMEtSequence"), postfix)
            lastCorrectedJetCollectionForNoPileUpPFMEt += postfix
                            
        if doSmearJets:
            process.load("RecoJets.Configuration.GenJetParticles_cff")
            metUncertaintySequence += process.genParticlesForJetsNoNu
            process.load("RecoJets.Configuration.RecoGenJets_cff")
            metUncertaintySequence += process.ak5GenJetsNoNu
            setattr(process, smearedUncorrectedJetCollection + postfix, cms.EDProducer("SmearedPFJetProducer",
                src = cms.InputTag(uncorrectedJetCollection),
                jetCorrLabel = cms.string(jetCorrLabelUpToL3),                                       
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
            metUncertaintySequence += getattr(process, smearedUncorrectedJetCollection + postfix)
            getattr(process, correctedJetCollection + postfix).src = cms.InputTag(smearedUncorrectedJetCollection + postfix)
        metUncertaintySequence += getattr(process, noPileUpPFMEtSequence + postfix)
        if doApplyChargedHadronSubtraction:
            self._addPATMEtProducer(process, metUncertaintySequence, 'noPileUpPFchsMEt' + postfix, 'patPFchsMetNoPileUp', collectionsToKeep, postfix)
        else:
            self._addPATMEtProducer(process, metUncertaintySequence, 'noPileUpPFMEt' + postfix, 'patPFMetNoPileUp', collectionsToKeep, postfix)

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
                    pfCandCollection, "ForNoPileUpPF%sMEt%s" % (chsLabel, postfix))
                modulePFCandidateToVertexAssociationShiftUp = process.pfCandidateToVertexAssociation.clone(
                    PFCandidateCollection = cms.InputTag(pfCandCollectionLeptonShiftUp)
                )
                modulePFCandidateToVertexAssociationShiftUpName = "pfCandidateToVertexAssociation%s%sUpForPileUpPF%sMEt%s" % (leptonCollection[0], leptonCollection[1], chsLabel, postfix)
                setattr(process, modulePFCandidateToVertexAssociationShiftUpName, modulePFCandidateToVertexAssociationShiftUp)
                metUncertaintySequence += modulePFCandidateToVertexAssociationShiftUp
                modulePFMEtDataLeptonShiftUp = getattr(process, noPileUpPFMEtData + postfix).clone(
                    srcPFCandidates = cms.InputTag(pfCandCollectionLeptonShiftUp),
                    srcPFCandToVertexAssociations = cms.InputTag(modulePFCandidateToVertexAssociationShiftUpName)
                )
                modulePFMEtDataLeptonShiftUpName = "%s%s%sUp%s" % (noPileUpPFMEtData, leptonCollection[0], leptonCollection[1], postfix)
                setattr(process, modulePFMEtDataLeptonShiftUpName, modulePFMEtDataLeptonShiftUp)
                metUncertaintySequence += modulePFMEtDataLeptonShiftUp
                modulePFMEtLeptonShiftUp = getattr(process, noPileUpPFMEt + postfix).clone(
                    srcMVAMEtData = cms.InputTag(modulePFMEtDataLeptonShiftUpName),
                    srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(
                      shiftedParticleCollections, leptonCollection[2], '%s%sUp' % (leptonCollection[2], leptonCollection[1]), postfix = postfix))
                )
                modulePFMEtLeptonShiftUpName = "%s%s%sUp%s" % (noPileUpPFMEt, leptonCollection[0], leptonCollection[1], postfix)
                setattr(process, modulePFMEtLeptonShiftUpName, modulePFMEtLeptonShiftUp)
                metUncertaintySequence += modulePFMEtLeptonShiftUp
                self._addPATMEtProducer(process, metUncertaintySequence,
                                        modulePFMEtLeptonShiftUpName, '%s%s%sUp' % (patPFMetNoPileUp, leptonCollection[0], leptonCollection[1]), collectionsToKeep, postfix)
                modulePFCandidateToVertexAssociationShiftDown = modulePFCandidateToVertexAssociationShiftUp.clone(
                    PFCandidateCollection = cms.InputTag(pfCandCollectionLeptonShiftDown)
                )
                modulePFCandidateToVertexAssociationShiftDownName = "pfCandidateToVertexAssociation%s%sDownForPileUpPF%sMEt%s" % (leptonCollection[0], leptonCollection[1], chsLabel, postfix)
                modulePFCandidateToVertexAssociationShiftDownName += postfix
                setattr(process, modulePFCandidateToVertexAssociationShiftDownName, modulePFCandidateToVertexAssociationShiftDown)
                metUncertaintySequence += modulePFCandidateToVertexAssociationShiftDown                
                modulePFMEtDataLeptonShiftDown = getattr(process, noPileUpPFMEtData + postfix).clone(
                    srcPFCandidates = cms.InputTag(pfCandCollectionLeptonShiftDown),
                    srcPFCandToVertexAssociations = cms.InputTag(modulePFCandidateToVertexAssociationShiftDownName)
                )
                modulePFMEtDataLeptonShiftDownName = "%s%s%sDown%s" % (noPileUpPFMEtData, leptonCollection[0], leptonCollection[1], postfix)
                setattr(process, modulePFMEtDataLeptonShiftDownName, modulePFMEtDataLeptonShiftDown)
                metUncertaintySequence += modulePFMEtDataLeptonShiftDown
                modulePFMEtLeptonShiftDown = getattr(process, noPileUpPFMEt + postfix).clone(
                    srcMVAMEtData = cms.InputTag(modulePFMEtDataLeptonShiftDownName),
                    srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(
                      shiftedParticleCollections, leptonCollection[2], '%s%sDown' % (leptonCollection[2], leptonCollection[1]), postfix = postfix))
                )
                modulePFMEtLeptonShiftDownName = "%s%s%sDown%s" % (noPileUpPFMEt, leptonCollection[0], leptonCollection[1], postfix)
                setattr(process, modulePFMEtLeptonShiftDownName, modulePFMEtLeptonShiftDown)
                metUncertaintySequence += modulePFMEtLeptonShiftDown
                self._addPATMEtProducer(process, metUncertaintySequence,
                                        modulePFMEtLeptonShiftDownName, '%s%s%sDown' % (patPFMetNoPileUp, leptonCollection[0], leptonCollection[1]), collectionsToKeep, postfix)

        if self._isValidInputTag(shiftedParticleCollections['jetCollection']):
            uncorrectedJetsEnUp = None
            correctedJetsEnUp = None
            if doApplyChargedHadronSubtraction:
                uncorrectedJetsEnUp = "uncorrectedJetsEnUpForNoPileUpPFchsMEt" + postfix
                correctedJetsEnUp = "correctedJetsEnUpForNoPileUpPFchsMEt" + postfix
            else:
                uncorrectedJetsEnUp = "uncorrectedJetsEnUpForNoPileUpPFMEt" + postfix
                correctedJetsEnUp = "correctedJetsEnUpForNoPileUpPFMEt" + postfix
            setattr(process, uncorrectedJetsEnUp, cms.EDProducer("ShiftedPFJetProducer",
                src = cms.InputTag(uncorrectedJetCollection),
                jetCorrInputFileName = cms.FileInPath(jecUncertaintyFile),
                jetCorrUncertaintyTag = cms.string(jecUncertaintyTag),
                addResidualJES = cms.bool(True),
                jetCorrLabelUpToL3 = cms.string(jetCorrLabelUpToL3),
                jetCorrLabelUpToL3Res = cms.string(jetCorrLabelUpToL3Residual),
                shiftBy = cms.double(+1.*varyByNsigmas),
                ##verbosity = cms.int32(1)
            ))
            metUncertaintySequence += getattr(process, uncorrectedJetsEnUp)
            setattr(process, correctedJetsEnUp, getattr(process, uncorrectedJetsEnUp).clone(
                src = cms.InputTag(correctedJetCollection),
                addResidualJES = cms.bool(False)
            ))
            metUncertaintySequence += getattr(process, correctedJetsEnUp)
            puJetIdJetEnUp = "%sJetEnUp%s" % (puJetId, postfix)
            setattr(process, puJetIdJetEnUp, getattr(process, puJetId + postfix).clone(
                jets = cms.InputTag(correctedJetsEnUp)
            ))
            metUncertaintySequence += getattr(process, puJetIdJetEnUp)
            noPileUpPFMEtDataJetEnUp = "%sJetEnUp%s" % (noPileUpPFMEtData, postfix)
            setattr(process, noPileUpPFMEtDataJetEnUp, getattr(process, noPileUpPFMEtData + postfix).clone(
                srcJets = cms.InputTag(correctedJetsEnUp),
                srcJetIds = cms.InputTag(puJetIdJetEnUp, 'fullId')
            ))
            metUncertaintySequence += getattr(process, noPileUpPFMEtDataJetEnUp)
            noPileUpPFMEtJetEnUp = "%sJetEnUp%s" % (noPileUpPFMEt, postfix)
            setattr(process, noPileUpPFMEtJetEnUp, getattr(process, noPileUpPFMEt + postfix).clone(
                srcMVAMEtData = cms.InputTag(noPileUpPFMEtDataJetEnUp),
                srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections, postfix = postfix))
            ))
            metUncertaintySequence += getattr(process, noPileUpPFMEtJetEnUp)
            self._addPATMEtProducer(process, metUncertaintySequence,
                                    noPileUpPFMEtJetEnUp, "%sJetEnUp" % patPFMetNoPileUp, collectionsToKeep, postfix)
            uncorrectedJetsEnDown = uncorrectedJetsEnUp.replace("JetsEnUp", "JetsEnDown")
            correctedJetsEnDown = correctedJetsEnUp.replace("JetsEnUp", "JetsEnDown")
            setattr(process, uncorrectedJetsEnDown, getattr(process, uncorrectedJetsEnUp).clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            ))
            metUncertaintySequence += getattr(process, uncorrectedJetsEnDown)
            setattr(process, correctedJetsEnDown, getattr(process, correctedJetsEnUp).clone(
                shiftBy = cms.double(-1.*varyByNsigmas)
            ))
            metUncertaintySequence += getattr(process, correctedJetsEnDown)
            puJetIdJetEnDown = "%sJetEnDown%s" % (puJetId, postfix)
            setattr(process, puJetIdJetEnDown, getattr(process, puJetIdJetEnUp).clone(
                jets = cms.InputTag(correctedJetsEnDown)
            ))
            metUncertaintySequence += getattr(process, puJetIdJetEnDown)
            noPileUpPFMEtDataJetEnDown = "%sJetEnDown%s" % (noPileUpPFMEtData, postfix)
            setattr(process, noPileUpPFMEtDataJetEnDown, getattr(process, noPileUpPFMEtData).clone(
                srcJets = cms.InputTag(correctedJetsEnDown),
                srcJetIds = cms.InputTag(puJetIdJetEnDown, 'fullId')
            ))
            metUncertaintySequence += getattr(process, noPileUpPFMEtDataJetEnDown)
            noPileUpPFMEtJetEnDown = "%sJetEnDown%s" % (noPileUpPFMEt, postfix)
            setattr(process, noPileUpPFMEtJetEnDown, getattr(process, noPileUpPFMEt).clone(
                srcMVAMEtData = cms.InputTag(noPileUpPFMEtDataJetEnDown),
                srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections, postfix = postfix))
            ))
            metUncertaintySequence += getattr(process, noPileUpPFMEtJetEnDown)
            self._addPATMEtProducer(process, metUncertaintySequence,
                                    noPileUpPFMEtJetEnDown, "%sJetEnDown" % patPFMetNoPileUp, collectionsToKeep, postfix)

            if hasattr(process, smearedUncorrectedJetCollection + postfix):
                setattr(process, smearedCorrectedJetCollection + postfix, getattr(process, smearedUncorrectedJetCollection + postfix).clone(
                    src = cms.InputTag(correctedJetCollection + postfix),
                    jetCorrLabel = cms.string("")
                ))
                correctedJetsResUp = None
                if doApplyChargedHadronSubtraction:
                    correctedJetsResUp = "correctedJetsResUpForNoPileUpPFchsMEt" + postfix
                else:
                    correctedJetsResUp = "correctedJetsResUpForNoPileUpPFMEt" + postfix
                setattr(process, correctedJetsResUp, getattr(process, smearedCorrectedJetCollection).clone(
                    shiftBy = cms.double(-1.*varyByNsigmas)
                ))
                metUncertaintySequence += getattr(process, correctedJetsResUp)
                puJetIdJetResUp = "%sJetResUp%s" % (puJetId, postfix)                
                setattr(process, puJetIdJetResUp, getattr(process, puJetId).clone(
                    jets = cms.InputTag(correctedJetsResUp)
                ))
                metUncertaintySequence += getattr(process, puJetIdJetResUp)
                noPileUpPFMEtDataJetResUp = "%sJetResUp%s" % (noPileUpPFMEtData, postfix)
                setattr(process, noPileUpPFMEtDataJetResUp, getattr(process, noPileUpPFMEtData).clone(
                    srcJets = cms.InputTag(correctedJetsResUp),
                    srcJetIds = cms.InputTag(puJetIdJetResUp, 'fullId')
                ))
                metUncertaintySequence += getattr(process, noPileUpPFMEtDataJetResUp)
                noPileUpPFMEtJetResUp = "%sJetResUp%s" % (noPileUpPFMEt, postfix)
                setattr(process, noPileUpPFMEtJetResUp, getattr(process, noPileUpPFMEt).clone(
                    srcMVAMEtData = cms.InputTag(noPileUpPFMEtDataJetResUp),
                    srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections, postfix = postfix))
                ))
                metUncertaintySequence += getattr(process, noPileUpPFMEtJetResUp)
                self._addPATMEtProducer(process, metUncertaintySequence,
                                        noPileUpPFMEtJetResUp, "%sJetResUp" % patPFMetNoPileUp, collectionsToKeep, postfix)
                correctedJetsResDown = correctedJetsResUp.replace("JetsResUp", "JetsResDown")
                setattr(process, correctedJetsResDown, getattr(process, smearedCorrectedJetCollection).clone(
                    shiftBy = cms.double(+1.*varyByNsigmas)
                ))
                metUncertaintySequence += getattr(process, correctedJetsResDown)
                puJetIdJetResDown = "%sJetResDown%s" % (puJetId, postfix)                
                setattr(process, puJetIdJetResDown, getattr(process, puJetId).clone(
                    jets = cms.InputTag(correctedJetsResDown)
                ))
                metUncertaintySequence += getattr(process, puJetIdJetResDown)
                noPileUpPFMEtDataJetResDown = "%sJetResDown%s" % (noPileUpPFMEtData, postfix)
                setattr(process, noPileUpPFMEtDataJetResDown, getattr(process, noPileUpPFMEtData).clone(
                    srcJets = cms.InputTag(correctedJetsResDown),
                    srcJetIds = cms.InputTag(puJetIdJetResDown, 'fullId')
                ))
                metUncertaintySequence += getattr(process, noPileUpPFMEtDataJetResDown)
                noPileUpPFMEtJetResDown = "%sJetResDown%s" % (noPileUpPFMEt, postfix)
                setattr(process, noPileUpPFMEtJetResDown, getattr(process, noPileUpPFMEt).clone(
                    srcMVAMEtData = cms.InputTag(noPileUpPFMEtDataJetResDown),
                    srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections, postfix = postfix))
                ))
                metUncertaintySequence += getattr(process, noPileUpPFMEtJetResDown)
                self._addPATMEtProducer(process, metUncertaintySequence,
                                        noPileUpPFMEtJetResDown, "%sJetResDown" % patPFMetNoPileUp, collectionsToKeep, postfix)

            pfCandsUnclusteredEnUp = None
            pfCandidateToVertexAssociationUnclusteredEnUp = None
            pfMETcorrType0UnclusteredEnUp = None
            if doApplyChargedHadronSubtraction:
                pfCandsUnclusteredEnUp = "pfCandsUnclusteredEnUpForNoPileUpPFchsMEt" + postfix
                pfCandidateToVertexAssociationUnclusteredEnUp = "pfCandidateToVertexAssociationUnclusteredEnUpForNoPileUpPFchsMEt" + postfix
                pfMETcorrType0UnclusteredEnUp = "pfMETcorrType0UnclusteredEnUpForNoPileUpPFchsMEt" + postfix
            else:
                pfCandsUnclusteredEnUp ="pfCandsUnclusteredEnUpForNoPileUpPFMEt" + postfix
                pfCandidateToVertexAssociationUnclusteredEnUp = "pfCandidateToVertexAssociationUnclusteredEnUpForNoPileUpPFMEt" + postfix
                pfMETcorrType0UnclusteredEnUp = "pfMETcorrType0UnclusteredEnUpForNoPileUpPFMEt" + postfix
            setattr(process, pfCandsUnclusteredEnUp, cms.EDProducer("ShiftedPFCandidateProducerForNoPileUpPFMEt",
                srcPFCandidates = cms.InputTag('particleFlow'),
                srcJets = cms.InputTag(correctedJetCollection),
                jetCorrInputFileName = cms.FileInPath(jecUncertaintyFile),
                jetCorrUncertaintyTag = cms.string(jecUncertaintyTag),
                minJetPt = cms.double(20.0),
                shiftBy = cms.double(+1.*varyByNsigmas),
                unclEnUncertainty = cms.double(0.10)
            ))
            metUncertaintySequence += getattr(process, pfCandsUnclusteredEnUp)
            setattr(process, pfCandidateToVertexAssociationUnclusteredEnUp, process.pfCandidateToVertexAssociation.clone(
                PFCandidateCollection = cms.InputTag(pfCandsUnclusteredEnUp)
            ))
            metUncertaintySequence += getattr(process, pfCandidateToVertexAssociationUnclusteredEnUp)
            setattr(process, pfMETcorrType0UnclusteredEnUp, getattr(process, "pfMETcorrType0" + postfix).clone(
                srcPFCandidateToVertexAssociations = cms.InputTag(pfCandidateToVertexAssociationUnclusteredEnUp)
            ))
            metUncertaintySequence += getattr(process, pfMETcorrType0UnclusteredEnUp)
            noPileUpPFMEtDataUnclusteredEnUp = "%sUnclusteredEnUp%s" % (noPileUpPFMEtData, postfix)
            setattr(process, noPileUpPFMEtDataUnclusteredEnUp, getattr(process, noPileUpPFMEtData + postfix).clone(
                srcPFCandidates = cms.InputTag(pfCandsUnclusteredEnUp),
                srcPFCandToVertexAssociations = cms.InputTag(pfCandidateToVertexAssociationUnclusteredEnUp)
            ))
            metUncertaintySequence += getattr(process, noPileUpPFMEtDataUnclusteredEnUp)
            noPileUpPFMEtUnclusteredEnUp = "%sUnclusteredEnUp%s" % (noPileUpPFMEt, postfix)
            setattr(process, noPileUpPFMEtUnclusteredEnUp, getattr(process, noPileUpPFMEt + postfix).clone(
                srcMVAMEtData = cms.InputTag(noPileUpPFMEtDataUnclusteredEnUp),
                srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections, postfix = postfix)),
                srcType0Correction = cms.InputTag(pfMETcorrType0UnclusteredEnUp)
            ))
            metUncertaintySequence += getattr(process, noPileUpPFMEtUnclusteredEnUp)
            self._addPATMEtProducer(process, metUncertaintySequence,
                                    noPileUpPFMEtUnclusteredEnUp, '%sUnclusteredEnUp' % patPFMetNoPileUp, collectionsToKeep, postfix)
            pfCandsUnclusteredEnDown = pfCandsUnclusteredEnUp.replace("UnclusteredEnUp", "UnclusteredEnDown")
            pfCandidateToVertexAssociationUnclusteredEnDown = pfCandidateToVertexAssociationUnclusteredEnUp.replace("UnclusteredEnUp", "UnclusteredEnDown")
            pfMETcorrType0UnclusteredEnDown = pfMETcorrType0UnclusteredEnUp.replace("UnclusteredEnUp", "UnclusteredEnDown")
            setattr(process, pfCandsUnclusteredEnDown, getattr(process, pfCandsUnclusteredEnUp).clone(
                shiftBy = cms.double(-1.*varyByNsigmas),
            ))
            metUncertaintySequence += getattr(process, pfCandsUnclusteredEnDown)
            setattr(process, pfCandidateToVertexAssociationUnclusteredEnDown, process.pfCandidateToVertexAssociation.clone(
                PFCandidateCollection = cms.InputTag(pfCandsUnclusteredEnDown)
            ))
            metUncertaintySequence += getattr(process, pfCandidateToVertexAssociationUnclusteredEnDown)
            setattr(process, pfMETcorrType0UnclusteredEnDown, getattr(process, "pfMETcorrType0" + postfix).clone(
                srcPFCandidateToVertexAssociations = cms.InputTag(pfCandidateToVertexAssociationUnclusteredEnDown)
            ))
            metUncertaintySequence += getattr(process, pfMETcorrType0UnclusteredEnDown)
            noPileUpPFMEtDataUnclusteredEnDown = "%sUnclusteredEnDown%s" % (noPileUpPFMEtData, postfix)
            setattr(process, noPileUpPFMEtDataUnclusteredEnDown, getattr(process, noPileUpPFMEtData + postfix).clone(
                srcPFCandidates = cms.InputTag(pfCandsUnclusteredEnDown),
                srcPFCandToVertexAssociations = cms.InputTag(pfCandidateToVertexAssociationUnclusteredEnDown)
            ))
            metUncertaintySequence += getattr(process, noPileUpPFMEtDataUnclusteredEnDown)
            noPileUpPFMEtUnclusteredEnDown = "%sUnclusteredEnDown%s" % (noPileUpPFMEt, postfix)
            setattr(process, noPileUpPFMEtUnclusteredEnDown, getattr(process, noPileUpPFMEt + postfix).clone(
                srcMVAMEtData = cms.InputTag(noPileUpPFMEtDataUnclusteredEnDown),
                srcLeptons = cms.VInputTag(self._getLeptonsForPFMEtInput(shiftedParticleCollections, postfix = postfix)),
                srcType0Correction = cms.InputTag(pfMETcorrType0UnclusteredEnDown)
            ))
            metUncertaintySequence += getattr(process, noPileUpPFMEtUnclusteredEnDown)
            self._addPATMEtProducer(process, metUncertaintySequence,
                                    noPileUpPFMEtUnclusteredEnDown, '%sUnclusteredEnDown' % patPFMetNoPileUp, collectionsToKeep, postfix)
    
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

        self.setParameter('doApplyChargedHadronSubtraction', doApplyChargedHadronSubtraction)
        self.setParameter('pfCandCollection', pfCandCollection)

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
        jetCorrPayloadName = self._parameters['jetCorrPayloadName'].value
        jetCorrLabelUpToL3 = self._parameters['jetCorrLabelUpToL3'].value
        jetCorrLabelUpToL3Res = self._parameters['jetCorrLabelUpToL3Res'].value
        jecUncertaintyFile = self._parameters['jecUncertaintyFile'].value
        jecUncertaintyTag = self._parameters['jecUncertaintyTag'].value
        varyByNsigmas = self._parameters['varyByNsigmas'].value
        addToPatDefaultSequence = self._parameters['addToPatDefaultSequence'].value
        outputModule = self._parameters['outputModule'].value
        postfix = self._parameters['postfix'].value

        if not hasattr(process, "pf%sNoPileUpMEtUncertaintySequence%s" % (chsLabel, postfix)):
            metUncertaintySequence = cms.Sequence()
            setattr(process, "pf%sNoPileUpMEtUncertaintySequence%s" % (chsLabel, postfix), metUncertaintySequence)
        metUncertaintySequence = getattr(process, "pf%sNoPileUpMEtUncertaintySequence%s" % (chsLabel, postfix))

        collectionsToKeep = []

        lastJetCollection = jetCollection.value()
        
        # smear jet energies to account for difference in jet resolutions between MC and Data
        # (cf. JME-10-014 PAS)        
        jetCollectionResUp = None
        jetCollectionResDown = None
        if doSmearJets:
            lastJetCollection = \
              self._addSmearedJets(process, lastJetCollection, [ "smeared", jetCollection.value(), "ForNoPileUpPF%sMEt" % chsLabel ],
                                   jetSmearFileName, jetSmearHistogram, varyByNsigmas,
                                   uncertaintySequence = metUncertaintySequence, postfix = postfix)
            jetCollectionResUp = \
              self._addSmearedJets(process, lastJetCollection, [ "smeared", jetCollection.value(), "ForNoPileUpPF%sMEtResUp" % chsLabel ],
                                   jetSmearFileName, jetSmearHistogram, varyByNsigmas, -1., 
                                   uncertaintySequence = metUncertaintySequence, postfix = postfix)
            collectionsToKeep.append(jetCollectionResUp)
            jetCollectionResDown = \
              self._addSmearedJets(process, lastJetCollection, [ "smeared", jetCollection.value(), "ForNoPileUpPF%sMEtResDown" % chsLabel ],
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
                                              jetCollection, None, lastJetCollection,
                                              jetCollectionResUp, jetCollectionResDown,
                                              jetCorrLabelUpToL3, jetCorrLabelUpToL3Res,
                                              jecUncertaintyFile, jecUncertaintyTag,
                                              varyByNsigmas,
                                              "ForNoPileUpPF%sMEt%s" % (chsLabel, postfix))
        setattr(process, "shiftedParticlesForNoPileUpPF%sMEtUncertainties%s" % (chsLabel, postfix), shiftedParticleSequence)    
        metUncertaintySequence += getattr(process, "shiftedParticlesForNoPileUpPF%sMEtUncertainties%s" % (chsLabel, postfix))
        collectionsToKeep.extend(addCollectionsToKeep)
        
        #--------------------------------------------------------------------------------------------    
        # propagate shifted particle energies to No-PU MET
        #--------------------------------------------------------------------------------------------

        self._addNoPileUpPFMEt(process, metUncertaintySequence,
                               shiftedParticleCollections, pfCandCollection,
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
