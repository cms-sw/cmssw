import FWCore.ParameterSet.Config as cms

# load modules for producing Type 1 / Type 1 + 2 corrections for reco::PFMET objects
from JetMETCorrections.Type1MET.correctionTermsPfMetType1Type2_cff import *

#from PhysicsTools.PatAlgos.producerLayer1.jetProducer_cfi import patJets


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

selectedPatJetsForMetT1T2Corr = cms.EDFilter("PATJetSelector",
    src = cms.InputTag('patJets'),
    cut = cms.string('abs(eta) < 9.9'),
    filter = cms.bool(False)
)

selectedPatJetsForMetT2Corr = cms.EDFilter("PATJetSelector",
    src = cms.InputTag('patJets'),
    cut = cms.string('abs(eta) > 9.9'),
    filter = cms.bool(False)
)
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# produce Type 1 + 2 MET corrections for pat::Jets of PF-type
patPFMetT1T2Corr = cms.EDProducer("PATPFJetMETcorrInputProducer",
    src = cms.InputTag('selectedPatJetsForMetT1T2Corr'),
    offsetCorrLabel = cms.InputTag("L1FastJet"),
    jetCorrLabel = cms.InputTag("L3Absolute"), # for MC
    jetCorrLabelRes = cms.InputTag("L2L3Residual"), # for Data automatic switch
    type1JetPtThreshold = cms.double(15.0),
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
patPFMetT1T2CorrSequence = cms.Sequence(selectedPatJetsForMetT1T2Corr*
                                        patPFMetT1T2Corr)

patPFMetT2Corr = patPFMetT1T2Corr.clone(
    src = cms.InputTag('selectedPatJetsForMetT2Corr')
)
patPFMetT2CorrSequence = cms.Sequence(patPFMetT2Corr)

#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# produce Type 0 MET corrections
from JetMETCorrections.Type1MET.correctionTermsPfMetType0PFCandidate_cff import *
patPFMetT0Corr = pfMETcorrType0.clone()
patPFMetT0CorrSequence = cms.Sequence(type0PFMEtCorrectionPFCandToVertexAssociation*patPFMetT0Corr)
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# produce Type xy MET corrections
from JetMETCorrections.Type1MET.pfMETmultShiftCorrections_cfi import *
#dummy module

patPFMetTxyCorr = pfMEtMultShiftCorr.clone()

patMultPhiCorrParams_Txy_50ns         = cms.VPSet( [pset for pset in multPhiCorrParams_Txy_50ns])
patMultPhiCorrParams_T0pcTxy_50ns     = cms.VPSet( [pset for pset in multPhiCorrParams_T0pcTxy_50ns])
patMultPhiCorrParams_T0pcT1Txy_50ns   = cms.VPSet( [pset for pset in multPhiCorrParams_T0pcT1Txy_50ns])
patMultPhiCorrParams_T0pcT1T2Txy_50ns = cms.VPSet( [pset for pset in multPhiCorrParams_T0pcT1T2Txy_50ns])
patMultPhiCorrParams_T1Txy_50ns       = cms.VPSet( [pset for pset in multPhiCorrParams_T1Txy_50ns])
patMultPhiCorrParams_T1T2Txy_50ns     = cms.VPSet( [pset for pset in multPhiCorrParams_T1T2Txy_50ns])
patMultPhiCorrParams_T1SmearTxy_50ns  = cms.VPSet( [pset for pset in multPhiCorrParams_T1Txy_50ns])
patMultPhiCorrParams_T1T2SmearTxy_50ns = cms.VPSet( [pset for pset in multPhiCorrParams_T1T2Txy_50ns])
patMultPhiCorrParams_T0pcT1SmearTxy_50ns = cms.VPSet( [pset for pset in multPhiCorrParams_T0pcT1Txy_50ns])
patMultPhiCorrParams_T0pcT1T2SmearTxy_50ns = cms.VPSet( [pset for pset in multPhiCorrParams_T0pcT1T2Txy_50ns])

patMultPhiCorrParams_Txy_25ns         = cms.VPSet( [pset for pset in multPhiCorrParams_Txy_25ns])
patMultPhiCorrParams_T0pcTxy_25ns     = cms.VPSet( [pset for pset in multPhiCorrParams_T0pcTxy_25ns])
patMultPhiCorrParams_T0pcT1Txy_25ns   = cms.VPSet( [pset for pset in multPhiCorrParams_T0pcT1Txy_25ns])
patMultPhiCorrParams_T0pcT1T2Txy_25ns = cms.VPSet( [pset for pset in multPhiCorrParams_T0pcT1T2Txy_25ns])
patMultPhiCorrParams_T1Txy_25ns       = cms.VPSet( [pset for pset in multPhiCorrParams_T1Txy_25ns])
patMultPhiCorrParams_T1T2Txy_25ns     = cms.VPSet( [pset for pset in multPhiCorrParams_T1T2Txy_25ns])
patMultPhiCorrParams_T1SmearTxy_25ns  = cms.VPSet( [pset for pset in multPhiCorrParams_T1Txy_25ns])
patMultPhiCorrParams_T1T2SmearTxy_25ns = cms.VPSet( [pset for pset in multPhiCorrParams_T1T2Txy_25ns])
patMultPhiCorrParams_T0pcT1SmearTxy_25ns = cms.VPSet( [pset for pset in multPhiCorrParams_T0pcT1Txy_25ns])
patMultPhiCorrParams_T0pcT1T2SmearTxy_25ns = cms.VPSet( [pset for pset in multPhiCorrParams_T0pcT1T2Txy_25ns])

#from Configuration.StandardSequences.Eras import eras
#eras.run2_50ns_specific.toModify(patPFMetTxyCorr, parameters=patMultPhiCorrParams_Txy_50ns )
#eras.run2_25ns_specific.toModify(patPFMetTxyCorr, parameters=patMultPhiCorrParams_Txy_25ns )

patPFMetTxyCorrSequence = cms.Sequence(patPFMetTxyCorr)

#--------------------------------------------------------------------------------
from RecoMET.METProducers.METSigParams_cfi import *
patSmearedJets = cms.EDProducer("SmearedPATJetProducer",
                   src = cms.InputTag("patJets"),
                   dRmaxGenJetMatch = cms.string('min(0.5, 0.1 + 0.3*exp(-0.05*(genJetPt - 10.)))'),
                   sigmaMaxGenJetMatch = cms.double(3.),
                   inputFileName = cms.FileInPath("PhysicsTools/PatUtils/data/pfJetResolutionMCtoDataCorrLUT.root"), #MM fixme
                   lutName = cms.string("pfJetResolutionMCtoDataCorrLUT"),
                   jetResolutions = METSignificance_params,
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
        'jecSetsAvailable && abs(energy - correctedP4("Uncorrected").energy) > (5.*min(energy, correctedP4("Uncorrected").energy))'
        ),
            skipRawJetPtThreshold = cms.double(10.), # GeV
            skipCorrJetPtThreshold = cms.double(1.e-2),
            verbosity = cms.int32(0)
)

selectedPatJetsForMetT1T2SmearCorr = cms.EDFilter("PATJetSelector",
    src = cms.InputTag('patSmearedJets'),
    cut = cms.string('abs(eta) < 9.9'),
    filter = cms.bool(False)
)

selectedPatJetsForMetT2SmearCorr = cms.EDFilter("PATJetSelector",
    src = cms.InputTag('patSmearedJets'),
    cut = cms.string('abs(eta) > 9.9'),
    filter = cms.bool(False)
)

patPFMetT1T2SmearCorr = patPFMetT1T2Corr.clone(
    src = cms.InputTag('selectedPatJetsForMetT1T2SmearCorr')
)

patPFMetT2SmearCorr = patPFMetT2Corr.clone(
    src = cms.InputTag('selectedPatJetsForMetT2SmearCorr')
)

patPFMetSmearCorrSequence = cms.Sequence(patSmearedJets*
                                         selectedPatJetsForMetT1T2SmearCorr*
                                         patPFMetT1T2SmearCorr)

#specific sequence for handling type2 correction with smeared jets
patPFMetT2SmearCorrSequence = cms.Sequence(patSmearedJets*
                                           selectedPatJetsForMetT1T2SmearCorr*
                                           selectedPatJetsForMetT2SmearCorr*
                                           patPFMetT1T2SmearCorr*
                                           patPFMetT2SmearCorr)

#--------------------------------------------------------------------------------
# use MET corrections to produce Type 1 / Type 1 + 2 corrected PFMET objects
patPFMetT1 = cms.EDProducer("CorrectedPATMETProducer",
    src = cms.InputTag('patPFMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('patPFMetT1T2Corr', 'type1'),
    ),
    applyType2Corrections = cms.bool(False)
)

patPFMetT1T2 = patPFMetT1.clone()
patPFMetT1T2.srcCorrections.append( cms.InputTag('patPFMetT2Corr',   'type2') )
#--------------------------------------------------------------------------------
#extra modules for naming scheme
patPFMetT1Txy = patPFMetT1.clone()
patPFMetT1Txy.srcCorrections.append( cms.InputTag('patPFMetTxyCorr') )

patPFMetT0pcT1 = patPFMetT1.clone()
patPFMetT0pcT1.srcCorrections.append( cms.InputTag('patPFMetT0Corr') )

patPFMetT0pcT1Txy = patPFMetT0pcT1.clone()
patPFMetT0pcT1Txy.srcCorrections.append( cms.InputTag('patPFMetTxyCorr') )

patPFMetT1T2Txy = patPFMetT1T2.clone()
patPFMetT1T2Txy.srcCorrections.append( cms.InputTag('patPFMetTxyCorr') )

patPFMetT0pcT1T2 = patPFMetT1T2.clone()
patPFMetT0pcT1T2.srcCorrections.append( cms.InputTag('patPFMetT0Corr') )

patPFMetT0pcT1T2Txy = patPFMetT0pcT1T2.clone()
patPFMetT0pcT1T2Txy.srcCorrections.append( cms.InputTag('patPFMetTxyCorr') )


## smeared METs
patPFMetT1Smear = patPFMetT1.clone( srcCorrections = cms.VInputTag(
        cms.InputTag('patPFMetT1T2SmearCorr', 'type1') )
)

patPFMetT1T2Smear = patPFMetT1Smear.clone()
patPFMetT1T2Smear.srcCorrections.append( cms.InputTag('patPFMetT2SmearCorr',   'type2') )

patPFMetT1TxySmear = patPFMetT1Smear.clone()
patPFMetT1TxySmear.srcCorrections.append( cms.InputTag('patPFMetTxyCorr') )

patPFMetT0pcT1Smear = patPFMetT1Smear.clone()
patPFMetT0pcT1Smear.srcCorrections.append( cms.InputTag('patPFMetT0Corr') )

patPFMetT0pcT1TxySmear = patPFMetT0pcT1Smear.clone()
patPFMetT0pcT1TxySmear.srcCorrections.append( cms.InputTag('patPFMetTxyCorr') )

patPFMetT1T2TxySmear = patPFMetT1T2Smear.clone()
patPFMetT1T2TxySmear.srcCorrections.append( cms.InputTag('patPFMetTxyCorr') )

patPFMetT0pcT1T2Smear = patPFMetT1T2Smear.clone()
patPFMetT0pcT1T2Smear.srcCorrections.append( cms.InputTag('patPFMetT0Corr') )

patPFMetT0pcT1T2TxySmear = patPFMetT0pcT1T2Smear.clone()
patPFMetT0pcT1T2TxySmear.srcCorrections.append( cms.InputTag('patPFMetTxyCorr') )

#--------------------------------------------------------------------------------
# define sequence to run all modules
producePatPFMETCorrections = cms.Sequence(
    patPFMet
   * pfCandsNotInJetsForMetCorr
   * selectedPatJetsForMetT1T2Corr
   * selectedPatJetsForMetT2Corr
   * patPFMetT1T2Corr
   * patPFMetT2Corr
   * type0PFMEtCorrectionPFCandToVertexAssociation
   * patPFMetT0Corr
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
   * selectedPatJetsForMetT1T2Corr
   * selectedPatJetsForMetT2Corr
   * patPFMetT1T2Corr
   * patPFMetT2Corr
   * type0PFMEtCorrectionPFCandToVertexAssociation
   * patPFMetT0Corr
   * pfCandMETcorr
)
