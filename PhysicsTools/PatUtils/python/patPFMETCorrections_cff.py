import FWCore.ParameterSet.Config as cms

# load modules for producing Type 1 / Type 1 + 2 corrections for reco::PFMET objects
from JetMETCorrections.Type1MET.correctionTermsPfMetType1Type2_cff import *

#from PhysicsTools.PatAlgos.producerLayer1.jetProducer_cfi import patJets


#--------------------------------------------------------------------------------
# produce "raw" (uncorrected) pat::MET of PF-type
from PhysicsTools.PatAlgos.producersLayer1.metProducer_cfi import patMETs
patPFMet = patMETs.clone(
    metSource = 'pfMet'
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
import PhysicsTools.PatUtils.pfJetMETcorrInputProducerTPatJetPATJetCorrExtractor_cfi as _mod

patPFMetT1T2Corr = _mod.pfJetMETcorrInputProducerTPatJetPATJetCorrExtractor.clone(
    src             = 'selectedPatJetsForMetT1T2Corr',
    offsetCorrLabel = "L1FastJet",
    jetCorrLabel    = "L3Absolute",  # for MC
    jetCorrLabelRes = "L2L3Residual" # for Data automatic switch
)
patPFMetT1T2CorrTask = cms.Task(selectedPatJetsForMetT1T2Corr,
                                patPFMetT1T2Corr)

patPFMetT2Corr = patPFMetT1T2Corr.clone(
    src = 'selectedPatJetsForMetT2Corr'
)
patPFMetT2CorrTask = cms.Task(patPFMetT2Corr)

#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# produce Type 0 MET corrections
from JetMETCorrections.Type1MET.pfMETCorrectionType0_cfi import *

patPFMetT0Corr = pfMETcorrType0.clone()
patPFMetT0CorrTask = cms.Task(type0PFMEtCorrectionPFCandToVertexAssociationTask, patPFMetT0Corr)
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# produce Type xy MET corrections
import JetMETCorrections.Type1MET.pfMETmultShiftCorrections_cfi as _shiftMod
#dummy module

patPFMetTxyCorr = _shiftMod.pfMEtMultShiftCorr.clone()

patMultPhiCorrParams_Txy_50ns         = cms.VPSet( [pset for pset in _shiftMod.multPhiCorrParams_Txy_50ns])
patMultPhiCorrParams_T0pcTxy_50ns     = cms.VPSet( [pset for pset in _shiftMod.multPhiCorrParams_T0pcTxy_50ns])
patMultPhiCorrParams_T0pcT1Txy_50ns   = cms.VPSet( [pset for pset in _shiftMod.multPhiCorrParams_T0pcT1Txy_50ns])
patMultPhiCorrParams_T0pcT1T2Txy_50ns = cms.VPSet( [pset for pset in _shiftMod.multPhiCorrParams_T0pcT1T2Txy_50ns])
patMultPhiCorrParams_T1Txy_50ns       = cms.VPSet( [pset for pset in _shiftMod.multPhiCorrParams_T1Txy_50ns])
patMultPhiCorrParams_T1T2Txy_50ns     = cms.VPSet( [pset for pset in _shiftMod.multPhiCorrParams_T1T2Txy_50ns])
patMultPhiCorrParams_T1SmearTxy_50ns  = cms.VPSet( [pset for pset in _shiftMod.multPhiCorrParams_T1Txy_50ns])
patMultPhiCorrParams_T1T2SmearTxy_50ns = cms.VPSet( [pset for pset in _shiftMod.multPhiCorrParams_T1T2Txy_50ns])
patMultPhiCorrParams_T0pcT1SmearTxy_50ns = cms.VPSet( [pset for pset in _shiftMod.multPhiCorrParams_T0pcT1Txy_50ns])
patMultPhiCorrParams_T0pcT1T2SmearTxy_50ns = cms.VPSet( [pset for pset in _shiftMod.multPhiCorrParams_T0pcT1T2Txy_50ns])

patMultPhiCorrParams_Txy_25ns         = cms.VPSet( [pset for pset in _shiftMod.multPhiCorrParams_Txy_25ns])
patMultPhiCorrParams_T0pcTxy_25ns     = cms.VPSet( [pset for pset in _shiftMod.multPhiCorrParams_T0pcTxy_25ns])
patMultPhiCorrParams_T0pcT1Txy_25ns   = cms.VPSet( [pset for pset in _shiftMod.multPhiCorrParams_T0pcT1Txy_25ns])
patMultPhiCorrParams_T0pcT1T2Txy_25ns = cms.VPSet( [pset for pset in _shiftMod.multPhiCorrParams_T0pcT1T2Txy_25ns])
patMultPhiCorrParams_T1Txy_25ns       = cms.VPSet( [pset for pset in _shiftMod.multPhiCorrParams_T1Txy_25ns])
patMultPhiCorrParams_T1T2Txy_25ns     = cms.VPSet( [pset for pset in _shiftMod.multPhiCorrParams_T1T2Txy_25ns])
patMultPhiCorrParams_T1SmearTxy_25ns  = cms.VPSet( [pset for pset in _shiftMod.multPhiCorrParams_T1Txy_25ns])
patMultPhiCorrParams_T1T2SmearTxy_25ns = cms.VPSet( [pset for pset in _shiftMod.multPhiCorrParams_T1T2Txy_25ns])
patMultPhiCorrParams_T0pcT1SmearTxy_25ns = cms.VPSet( [pset for pset in _shiftMod.multPhiCorrParams_T0pcT1Txy_25ns])
patMultPhiCorrParams_T0pcT1T2SmearTxy_25ns = cms.VPSet( [pset for pset in _shiftMod.multPhiCorrParams_T0pcT1T2Txy_25ns])

# Run2 UL MC XY(Type1 PFMET Phi) corrections
import JetMETCorrections.Type1MET.multPhiCorr_Run2_ULMC_cfi as multPhiCorrParams_Run2_ULMC
import JetMETCorrections.Type1MET.multPhiCorr_Run2_ULDATA_cfi as multPhiCorrParams_Run2_ULDATA
import JetMETCorrections.Type1MET.multPhiCorr_Puppi_Run2_ULMC_cfi as multPhiCorrParams_Puppi_Run2_ULMC
import JetMETCorrections.Type1MET.multPhiCorr_Puppi_Run2_ULDATA_cfi as multPhiCorrParams_Puppi_Run2_ULDATA

# PFMET XY corrections
patMultPhiCorrParams_ULMC2018 = multPhiCorrParams_Run2_ULMC.multPhiCorr_ULMC2018
patMultPhiCorrParams_ULMC2017 = multPhiCorrParams_Run2_ULMC.multPhiCorr_ULMC2017
patMultPhiCorrParams_ULMC2016preVFP = multPhiCorrParams_Run2_ULMC.multPhiCorr_ULMC2016preVFP
patMultPhiCorrParams_ULMC2016postVFP = multPhiCorrParams_Run2_ULMC.multPhiCorr_ULMC2016postVFP

patMultPhiCorrParams_ULDATA2018A = multPhiCorrParams_Run2_ULDATA.multPhiCorr_ULDATA2018A
patMultPhiCorrParams_ULDATA2018B = multPhiCorrParams_Run2_ULDATA.multPhiCorr_ULDATA2018B
patMultPhiCorrParams_ULDATA2018C = multPhiCorrParams_Run2_ULDATA.multPhiCorr_ULDATA2018C
patMultPhiCorrParams_ULDATA2018D = multPhiCorrParams_Run2_ULDATA.multPhiCorr_ULDATA2018D

patMultPhiCorrParams_ULDATA2017B = multPhiCorrParams_Run2_ULDATA.multPhiCorr_ULDATA2017B
patMultPhiCorrParams_ULDATA2017C = multPhiCorrParams_Run2_ULDATA.multPhiCorr_ULDATA2017C
patMultPhiCorrParams_ULDATA2017D = multPhiCorrParams_Run2_ULDATA.multPhiCorr_ULDATA2017D
patMultPhiCorrParams_ULDATA2017E = multPhiCorrParams_Run2_ULDATA.multPhiCorr_ULDATA2017E
patMultPhiCorrParams_ULDATA2017F = multPhiCorrParams_Run2_ULDATA.multPhiCorr_ULDATA2017F

patMultPhiCorrParams_ULDATA2016preVFPB = multPhiCorrParams_Run2_ULDATA.multPhiCorr_ULDATA2016preVFPB
patMultPhiCorrParams_ULDATA2016preVFPC = multPhiCorrParams_Run2_ULDATA.multPhiCorr_ULDATA2016preVFPC
patMultPhiCorrParams_ULDATA2016preVFPD = multPhiCorrParams_Run2_ULDATA.multPhiCorr_ULDATA2016preVFPD
patMultPhiCorrParams_ULDATA2016preVFPE = multPhiCorrParams_Run2_ULDATA.multPhiCorr_ULDATA2016preVFPE
patMultPhiCorrParams_ULDATA2016preVFPF = multPhiCorrParams_Run2_ULDATA.multPhiCorr_ULDATA2016preVFPF

patMultPhiCorrParams_ULDATA2016postVFPF = multPhiCorrParams_Run2_ULDATA.multPhiCorr_ULDATA2016postVFPF
patMultPhiCorrParams_ULDATA2016postVFPG = multPhiCorrParams_Run2_ULDATA.multPhiCorr_ULDATA2016postVFPG
patMultPhiCorrParams_ULDATA2016postVFPH = multPhiCorrParams_Run2_ULDATA.multPhiCorr_ULDATA2016postVFPH

# PuppiMET XY corrections
patMultPhiCorrParams_Puppi_ULMC2018 = multPhiCorrParams_Puppi_Run2_ULMC.multPhiCorr_Puppi_ULMC2018
patMultPhiCorrParams_Puppi_ULMC2017 = multPhiCorrParams_Puppi_Run2_ULMC.multPhiCorr_Puppi_ULMC2017
patMultPhiCorrParams_Puppi_ULMC2016preVFP = multPhiCorrParams_Puppi_Run2_ULMC.multPhiCorr_Puppi_ULMC2016preVFP
patMultPhiCorrParams_Puppi_ULMC2016postVFP = multPhiCorrParams_Puppi_Run2_ULMC.multPhiCorr_Puppi_ULMC2016postVFP

patMultPhiCorrParams_Puppi_ULDATA2018A = multPhiCorrParams_Puppi_Run2_ULDATA.multPhiCorr_Puppi_ULDATA2018A
patMultPhiCorrParams_Puppi_ULDATA2018B = multPhiCorrParams_Puppi_Run2_ULDATA.multPhiCorr_Puppi_ULDATA2018B
patMultPhiCorrParams_Puppi_ULDATA2018C = multPhiCorrParams_Puppi_Run2_ULDATA.multPhiCorr_Puppi_ULDATA2018C
patMultPhiCorrParams_Puppi_ULDATA2018D = multPhiCorrParams_Puppi_Run2_ULDATA.multPhiCorr_Puppi_ULDATA2018D

patMultPhiCorrParams_Puppi_ULDATA2017B = multPhiCorrParams_Puppi_Run2_ULDATA.multPhiCorr_Puppi_ULDATA2017B
patMultPhiCorrParams_Puppi_ULDATA2017C = multPhiCorrParams_Puppi_Run2_ULDATA.multPhiCorr_Puppi_ULDATA2017C
patMultPhiCorrParams_Puppi_ULDATA2017D = multPhiCorrParams_Puppi_Run2_ULDATA.multPhiCorr_Puppi_ULDATA2017D
patMultPhiCorrParams_Puppi_ULDATA2017E = multPhiCorrParams_Puppi_Run2_ULDATA.multPhiCorr_Puppi_ULDATA2017E
patMultPhiCorrParams_Puppi_ULDATA2017F = multPhiCorrParams_Puppi_Run2_ULDATA.multPhiCorr_Puppi_ULDATA2017F

patMultPhiCorrParams_Puppi_ULDATA2016preVFPB = multPhiCorrParams_Puppi_Run2_ULDATA.multPhiCorr_Puppi_ULDATA2016preVFPB
patMultPhiCorrParams_Puppi_ULDATA2016preVFPC = multPhiCorrParams_Puppi_Run2_ULDATA.multPhiCorr_Puppi_ULDATA2016preVFPC
patMultPhiCorrParams_Puppi_ULDATA2016preVFPD = multPhiCorrParams_Puppi_Run2_ULDATA.multPhiCorr_Puppi_ULDATA2016preVFPD
patMultPhiCorrParams_Puppi_ULDATA2016preVFPE = multPhiCorrParams_Puppi_Run2_ULDATA.multPhiCorr_Puppi_ULDATA2016preVFPE
patMultPhiCorrParams_Puppi_ULDATA2016preVFPF = multPhiCorrParams_Puppi_Run2_ULDATA.multPhiCorr_Puppi_ULDATA2016preVFPF

patMultPhiCorrParams_Puppi_ULDATA2016postVFPF = multPhiCorrParams_Puppi_Run2_ULDATA.multPhiCorr_Puppi_ULDATA2016postVFPF
patMultPhiCorrParams_Puppi_ULDATA2016postVFPG = multPhiCorrParams_Puppi_Run2_ULDATA.multPhiCorr_Puppi_ULDATA2016postVFPG
patMultPhiCorrParams_Puppi_ULDATA2016postVFPH = multPhiCorrParams_Puppi_Run2_ULDATA.multPhiCorr_Puppi_ULDATA2016postVFPH

patPFMetTxyCorrTask = cms.Task(patPFMetTxyCorr)

#--------------------------------------------------------------------------------
from RecoMET.METProducers.METSigParams_cfi import *
patSmearedJets = cms.EDProducer("SmearedPATJetProducer",
    src = cms.InputTag("patJets"),

    enabled = cms.bool(True),  # If False, no smearing is performed

    rho = cms.InputTag("fixedGridRhoFastjetAll"),

    skipGenMatching = cms.bool(False),  # If True, always skip gen jet matching and smear jet with a random gaussian

    # Resolution and scale factors source.
    # Can be either from GT or text files
    # For GT: only 'algo' must be set
    # For text files: both 'resolutionFile' and 'scaleFactorFile' must point to valid files

    # Read from GT
    algopt = cms.string('AK4PFchs_pt'),
    algo = cms.string('AK4PFchs'),

    # Or from text files
    #resolutionFile = cms.FileInPath('path/to/resolution_file.txt'),
    #scaleFactorFile = cms.FileInPath('path/to/scale_factor_file.txt'),

    # Gen jet matching
    genJets = cms.InputTag("ak4GenJetsNoNu"),
    dRMax = cms.double(0.2),  # = cone size (0.4) / 2
    dPtMaxFactor = cms.double(3),  # dPt < 3 * resolution

    # Systematic variation
    # 0: Nominal
    # -1: -1 sigma (down variation)
    # 1: +1 sigma (up variation)
    variation = cms.int32(0),  # If not specified, default to 0

    seed = cms.uint32(37428479),  # If not specified, default to 37428479
    useDeterministicSeed = cms.bool(True),

    debug = cms.untracked.bool(False)
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
    src = 'selectedPatJetsForMetT1T2SmearCorr'
)

patPFMetT2SmearCorr = patPFMetT2Corr.clone(
    src = 'selectedPatJetsForMetT2SmearCorr'
)

patPFMetSmearCorrTask = cms.Task(patSmearedJets,
                                 selectedPatJetsForMetT1T2SmearCorr,
                                 patPFMetT1T2SmearCorr)

#specific sequence for handling type2 correction with smeared jets
patPFMetT2SmearCorrTask = cms.Task(patSmearedJets,
                                   selectedPatJetsForMetT1T2SmearCorr,
                                   selectedPatJetsForMetT2SmearCorr,
                                   patPFMetT1T2SmearCorr,
                                   patPFMetT2SmearCorr)

#--------------------------------------------------------------------------------
# use MET corrections to produce Type 1 / Type 1 + 2 corrected PFMET objects
patPFMetT1 = cms.EDProducer("CorrectedPATMETProducer",
    src = cms.InputTag('patPFMet'),
    srcCorrections = cms.VInputTag(
        cms.InputTag('patPFMetT1T2Corr', 'type1'),
    ),
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
patPFMetT1Smear = patPFMetT1.clone( 
    srcCorrections = ['patPFMetT1T2SmearCorr:type1']
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
producePatPFMETCorrectionsTask = cms.Task(
    patPFMet,
    pfCandsNotInJetsForMetCorr,
    selectedPatJetsForMetT1T2Corr,
    selectedPatJetsForMetT2Corr,
    patPFMetT1T2Corr,
    patPFMetT2Corr,
    type0PFMEtCorrectionPFCandToVertexAssociationTask,
    patPFMetT0Corr,
    pfCandMETcorr,
    patPFMetT1,
    patPFMetT1T2,
    patPFMetT0pcT1,
    patPFMetT0pcT1T2
)
#--------------------------------------------------------------------------------

#
# define special sequence for PAT runType1uncertainty tool
# only preliminary modules processed
# pat met producer modules cloned accordingly to what is needed
producePatPFMETCorrectionsUncTask = cms.Task(
    patPFMet,
    pfCandsNotInJetsForMetCorr,
    selectedPatJetsForMetT1T2Corr,
    selectedPatJetsForMetT2Corr,
    patPFMetT1T2Corr,
    patPFMetT2Corr,
    type0PFMEtCorrectionPFCandToVertexAssociationTask,
    patPFMetT0Corr,
    pfCandMETcorr
)
