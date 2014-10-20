import FWCore.ParameterSet.Config as cms

# load modules for producing Type 1 / Type 1 + 2 corrections for reco::PFMET objects
from JetMETCorrections.Type1MET.correctionTermsPfMetType1Type2_cff import *

#--------------------------------------------------------------------------------
# produce "raw" (uncorrected) pat::MET of PF-type
from PhysicsTools.PatAlgos.producersLayer1.metProducer_cfi import patMETs
patPFMet = patMETs.clone(
    metSource = cms.InputTag('pfMet'),
    addMuonCorrections = cms.bool(False),
    genMETSource = cms.InputTag('genMetTrue')
)
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# select collection of pat::Jets entering Type 1 + 2 MET corrections
#

selectedPatJetsForMETtype1p2Corr = cms.EDFilter("PATJetSelector",
    src = cms.InputTag('patJets'),
    cut = cms.string('abs(eta) < 9.9'),
    filter = cms.bool(False)
)

selectedPatJetsForMETtype2Corr = cms.EDFilter("PATJetSelector",
    src = cms.InputTag('patJets'),
    cut = cms.string('abs(eta) > 9.9'),
    filter = cms.bool(False)
)
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# produce Type 1 + 2 MET corrections for pat::Jets of PF-type
patPFJetMETtype1p2Corr = cms.EDProducer("PATPFJetMETcorrInputProducer",
    src = cms.InputTag('selectedPatJetsForMETtype1p2Corr'),
    offsetCorrLabel = cms.InputTag("L1FastJet"),
    jetCorrLabel = cms.InputTag("L3Absolute"), # NOTE: use "L3Absolute" for MC / "L2L3Residual" for Data
    type1JetPtThreshold = cms.double(10.0),
    type2ResidualCorrLabel = cms.InputTag(""),
    type2ResidualCorrEtaMax = cms.double(9.9),
    type2ExtraCorrFactor = cms.double(1.),
    type2ResidualCorrOffset = cms.double(0.),
    isMC = cms.bool(False), # CV: only used to decide whether to apply "unclustered energy" calibration to MC or Data
    skipEM = cms.bool(True),
    skipEMfractionThreshold = cms.double(0.90),
    skipMuons = cms.bool(True),
    skipMuonSelection = cms.string("isGlobalMuon | isStandAloneMuon")
)

patPFJetMETtype2Corr = patPFJetMETtype1p2Corr.clone(
    src = cms.InputTag('selectedPatJetsForMETtype2Corr')
)
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# produce Type 0 MET corrections
from JetMETCorrections.Type1MET.correctionTermsPfMetType0PFCandidate_cff import *
patPFMETtype0Corr = pfMETcorrType0.clone()
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# use MET corrections to produce Type 1 / Type 1 + 2 corrected PFMET objects
patPFMetT1 = cms.EDProducer("CorrectedPATMETProducer",
    src = cms.InputTag('patPFMet'),
    applyType1Corrections = cms.bool(True),
    srcType1Corrections = cms.VInputTag(
        cms.InputTag('patPFJetMETtype1p2Corr', 'type1'),
#        cms.InputTag('patPFMETtype0Corr')
    ),
    applyType2Corrections = cms.bool(False)
)

patPFMetT1T2 = cms.EDProducer("CorrectedPATMETProducer",
    src = cms.InputTag('patPFMet'),
    applyType1Corrections = cms.bool(True),
    srcType1Corrections = cms.VInputTag(
        cms.InputTag('patPFJetMETtype1p2Corr', 'type1'),
#        cms.InputTag('patPFMETtype0Corr')
    ),
    applyType2Corrections = cms.bool(True),
    srcUnclEnergySums = cms.VInputTag(
        cms.InputTag('patPFJetMETtype1p2Corr', 'type2' ),
        cms.InputTag('patPFJetMETtype2Corr',   'type2' ),
        cms.InputTag('patPFJetMETtype1p2Corr', 'offset'),
        cms.InputTag('pfCandMETcorr')
    ),
    type2CorrFormula = cms.string("A"),
    type2CorrParameter = cms.PSet(
        A = cms.double(1.4)
    )
)
#--------------------------------------------------------------------------------
#extra modules for naming scheme
patPFMetT1Txy = patPFMetT1.clone()
patPFMetT0pcT1 = patPFMetT1.clone()
patPFMetT0pcT1.srcType1Corrections.append( cms.InputTag('patPFMETtype0Corr') )

patPFMetT0pcT1Txy = patPFMetT1.clone()
patPFMetT0pcT1Txy.srcType1Corrections.append( cms.InputTag('patPFMETtype0Corr') )

patPFMetT1T2Txy = patPFMetT1T2.clone()
patPFMetT0pcT1T2 = patPFMetT1T2.clone()
patPFMetT0pcT1T2.srcType1Corrections.append( cms.InputTag('patPFMETtype0Corr') )

patPFMetT0pcT1T2Txy = patPFMetT1T2.clone()
patPFMetT0pcT1T2Txy.srcType1Corrections.append( cms.InputTag('patPFMETtype0Corr') )

#--------------------------------------------------------------------------------
# define sequence to run all modules
producePatPFMETCorrections = cms.Sequence(
    patPFMet
   * pfCandsNotInJetsForMetCorr
   * selectedPatJetsForMETtype1p2Corr
   * selectedPatJetsForMETtype2Corr
   * patPFJetMETtype1p2Corr
   * patPFJetMETtype2Corr
   * type0PFMEtCorrectionPFCandToVertexAssociation
   * patPFMETtype0Corr
   * pfCandMETcorr
   * patPFMetT1
   * patPFMetT1T2
   * patPFMetT0pcT1
   * patPFMetT0pcT1T2
)
#--------------------------------------------------------------------------------

#
# define special sequence for PAT runType1uncertainty tool
# only preliminary modules processed
# pat met producer modules cloned accordingly to what is needed
producePatPFMETCorrectionsUnc = cms.Sequence(
    patPFMet
   * pfCandsNotInJetsForMetCorr
   * selectedPatJetsForMETtype1p2Corr
   * selectedPatJetsForMETtype2Corr
   * patPFJetMETtype1p2Corr
   * patPFJetMETtype2Corr
   * type0PFMEtCorrectionPFCandToVertexAssociation
   * patPFMETtype0Corr
   * pfCandMETcorr
)
