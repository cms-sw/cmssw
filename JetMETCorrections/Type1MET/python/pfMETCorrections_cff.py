import FWCore.ParameterSet.Config as cms

# load jet energy correction parameters
from JetMETCorrections.Configuration.JetCorrectionServices_cff import *

#--------------------------------------------------------------------------------
# produce rho parameters needed for L1FastJet corrections
from CommonTools.ParticleFlow.pfNoPileUp_cff import pfNoPileUpSequence

from RecoJets.JetProducers.kt4PFJets_cfi import kt4PFJets
kt6PFJets = kt4PFJets.clone(
    src = cms.InputTag('particleFlow'),
    rParam = cms.double(0.6),
    doAreaFastjet = cms.bool(True),
    doRhoFastjet = cms.bool(True),
    Rho_EtaMax = cms.double(2.5)
)

# CV: need to rerun 'ak5PFJets' module with jet area computation enabled,
#     since it has not been enabled per default in CMSSW_4_2_x
#    (if the jet area of 'ak5PFJets' is zero, the L1FastjetCorrector::correction function always returns 1.0)
from RecoJets.JetProducers.ak5PFJets_cfi import ak5PFJets
ak5PFJets.doAreaFastjet = cms.bool(True)
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# select PFCandidates for Type 2 MET correction:
#  (1) select PFCandidates ("unclustered energy") not within jets
from CommonTools.ParticleFlow.TopProjectors.pfNoJet_cfi import pfNoJet
pfCandsNotInJet = pfNoJet.clone(
    topCollection = cms.InputTag('ak5PFJets'),
    bottomCollection = cms.InputTag('particleFlow')
)
#  (2) select subset of PFCandidates corresponding to neutral hadrons or photons
from CommonTools.ParticleFlow.ParticleSelectors.pfAllChargedHadrons_cfi import pfAllChargedHadrons
from CommonTools.ParticleFlow.ParticleSelectors.pfAllElectrons_cfi import pfAllElectrons
from CommonTools.ParticleFlow.ParticleSelectors.pfAllMuons_cfi import pfAllMuons
from CommonTools.ParticleFlow.ParticleSelectors.pfAllNeutralHadrons_cfi import pfAllNeutralHadrons
from CommonTools.ParticleFlow.ParticleSelectors.pfAllPhotons_cfi import pfAllPhotons
pfType2CandPdgIds = []
#pfType2CandPdgIds.extend(pfAllChargedHadrons.pdgId.value())
#pfType2CandPdgIds.extend(pfAllElectrons.pdgId.value())
#pfType2CandPdgIds.extend(pfAllMuons.pdgId.value())
pfType2CandPdgIds.extend(pfAllNeutralHadrons.pdgId.value())
pfType2CandPdgIds.extend(pfAllPhotons.pdgId.value())

pfType2Cands = cms.EDFilter("PdgIdPFCandidateSelector",
    src = cms.InputTag('pfCandsNotInJet'),
    pdgId = cms.vint32(pfType2CandPdgIds)
)
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# produce Type 1 + 2 MET corrections for PFJets
pfJetMETcorr = cms.EDProducer("PFJetMETcorrInputProducer",
    src = cms.InputTag('ak5PFJets'),
    offsetCorrLabel = cms.string("ak5PFL1Fastjet"),
    jetCorrLabel = cms.string("ak5PFL1FastL2L3"), # NOTE: use "ak5PFL1FastL2L3" for MC / "ak5PFL1FastL2L3Residual" for Data
    jetCorrEtaMax = cms.double(4.7),
    type1JetPtThreshold = cms.double(10.0),
    skipEM = cms.bool(True),
    skipEMfractionThreshold = cms.double(0.90),
    skipMuons = cms.bool(True),
    skipMuonSelection = cms.string("isGlobalMuon | isStandAloneMuon")
)                                         
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# produce Type 2 MET corrections for selected PFCandidates
pfCandMETcorr = cms.EDProducer("PFCandMETcorrInputProducer",
    src = cms.InputTag('pfType2Cands')
)   
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# use MET corrections to produce Type 1 / Type 1 + 2 corrected PFMET objects
pfType1CorrectedMet = cms.EDProducer("CorrectedPFMETProducer",
    src = cms.InputTag('pfMet'),
    applyType1Corrections = cms.bool(True),
    srcType1Corrections = cms.VInputTag(
        cms.InputTag('pfJetMETcorr', 'type1')
    ),
    applyType2Corrections = cms.bool(False)
)   

pfType1p2CorrectedMet = cms.EDProducer("CorrectedPFMETProducer",
    src = cms.InputTag('pfMet'),
    applyType1Corrections = cms.bool(True),
    srcType1Corrections = cms.VInputTag(
        cms.InputTag('pfJetMETcorr', 'type1')
    ),
    applyType2Corrections = cms.bool(True),
    srcUnclEnergySums = cms.VInputTag(
        cms.InputTag('pfJetMETcorr', 'type2'),
        cms.InputTag('pfJetMETcorr', 'offset'),
        cms.InputTag('pfCandMETcorr')                                    
    ),                              
    type2CorrFormula = cms.string("A"),
    type2CorrParameter = cms.PSet(
        A = cms.double(1.2)
    )
)   
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# define sequence to run all modules
producePFMETCorrections = cms.Sequence(
    pfNoPileUpSequence
   * kt6PFJets
   * ak5PFJets
   * pfCandsNotInJet
   * pfType2Cands
   * pfJetMETcorr
   * pfCandMETcorr
   * pfType1CorrectedMet
   * pfType1p2CorrectedMet
)
#--------------------------------------------------------------------------------
