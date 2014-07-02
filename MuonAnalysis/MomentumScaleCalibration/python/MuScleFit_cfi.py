# -*- coding: utf-8 -*-

import FWCore.ParameterSet.Config as cms

# This line is only necessary when running on fastSim
SimTracksCollection = cms.untracked.InputTag("famosSimHits"),
# This must be set to true when using events generated with Sherpa
Sherpa = cms.untracked.bool(True),

# This line allows to use the EDLooper or to loop by hand.
# All the necessary information is saved during the first loop so there is not need
# at this time to read again the events in successive iterations. Therefore by default
# for iterations > 1 the loops are done by hand, which means that the framework does
# not need to read all the events again. This is much faster.
# If you need to read the events in every iteration put this to false.
FastLoop = cms.untracked.bool(False),

# Set the probability file location. First looks in the ProbabilitiesFile path (absolute path)
ProbabilitiesFile = cms.untracked.string("/home/demattia/FSR/CMSSW_3_6_1_patch4/src/MuonAnalysis/MomentumScaleCalibration/test/Probs_merge.root"),
ProbabilitiesFileInPath = cms.untracked.string("MuonAnalysis/MomentumScaleCalibration/test/Probs_merge.root"),

# Name of the output files
OutputFileName = cms.untracked.string("MuScleFit.root"),
OutputGenInfoFileName = cms.untracked.string("genSimRecoPlots.root"),

debug = cms.untracked.int32(0),

# The following parameters can be used to filter events
TriggerResultsLabel = cms.untracked.string("TriggerResults"),
TriggerResultsProcess = cms.untracked.string("HLT"),
# TriggerPath: "" = No trigger requirements, "All" = No specific path
TriggerPath = cms.untracked.string("All"),
# Negate the result of the trigger
NegateTrigger = cms.untracked.bool(False),

# Decide whether to discard empty events or not
SaveAllToTree = cms.untracked.bool(False),

# Pile-Up related info
PileUpSummaryInfo = cms.untracked.InputTag("addPileupInfo"),
PrimaryVertexCollection = cms.untracked.InputTag("offlinePrimaryVertices"),

PATmuons = cms.untracked.bool(False),
GenParticlesName = cms.untracked.string("genParticles"),

# Use the probability file or not. If not it will perform a simpler selection taking the muon pair with
# invariant mass closer to the pdf value and will crash if some fit is attempted.
UseProbsFile = cms.untracked.bool(True),

# This must be set to true if using events generated with Sherpa
Sherpa = cms.untracked.bool(False),

# Use the rapidity bins or the single histogram for the Z
RapidityBinsForZ = cms.untracked.bool(True),

# Set the cuts on muons to be used in the fit
SeparateRanges = cms.untracked.bool(True),
MaxMuonPt = cms.untracked.double(100000000.),
MinMuonPt = cms.untracked.double(0.),
MinMuonEtaFirstRange = cms.untracked.double(-6.),
MaxMuonEtaFirstRange = cms.untracked.double(6.),
MinMuonEtaSecondRange = cms.untracked.double(-100.),
MaxMuonEtaSecondRange = cms.untracked.double(100.),
DeltaPhiMinCut = cms.untracked.double(0.),
DeltaPhiMaxCut = cms.untracked.double(100.),

# Produce additional plots on the mass resolution
DebugMassResol = cms.untracked.bool(False),

# Normalize the likelihood value with the number of events. If true (default), the error is also scaled
# with the appropriate factor to keep into account the different normalization of the likelihood.
NormalizeLikelihoodByEventNumber = cms.untracked.bool(True),

# Additional settings for Minuit
StartWithSimplex = cms.untracked.bool(True),
# This can be very time consuming depending on the number of events
ComputeMinosErrors = cms.untracked.bool(False),
MinimumShapePlots = cms.untracked.bool(True),
