import FWCore.ParameterSet.Config as cms

#
# Running the HLT table directly from Fast Simulation 
#
#--- Dummy replacements of HLT modules ---#
import FastSimulation.HighLevelTrigger.DummyModule_cfi

hltHfreco = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltHoreco = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltHbhereco = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltEcalDetIdToBeRecovered = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()

hltScalersRawToDigi = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()

hltEcalPreshowerDigis = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltEcalRawToRecHitFacility = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltESRawToRecHitFacility = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltEcalRegionalEtaFEDs = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltEcalRegionalEtaRecHit = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltEcalRegionalJetsFEDs = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltEcalRegionalJetsDigis = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltEcalRegionalJetsWeightUncalibRecHit = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltEcalRegionalJetsRecHitTmp = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltEcalRegionalMuonsFEDs = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltEcalRegionalMuonsDigis = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltEcalRegionalMuonsWeightUncalibRecHit = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltEcalRegionalMuonsRecHitTmp = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltEcalRegionalEgammaFEDs = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltEcalRegionalEgammaDigis = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltEcalRegionalEgammaWeightUncalibRecHit = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltEcalRegionalEgammaRecHitTmp = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltEcalRegionalTausFEDs = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltEcalRegionalTausDigis = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltEcalRegionalTausWeightUncalibRecHit = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltEcalRegionalTausRecHitTmp = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltEcalDigis = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltEcalWeightUncalibRecHit = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltHcalDigis = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltEcalRegionalRestFEDs = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltEcalRegionalESRestFEDs = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltEcalRegionalRestDigis = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltEcalRegionalRestWeightUncalibRecHit = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltEcalRegionalRestRecHitTmp = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltL3SingleTauPixelSeeds = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltL3SingleTauPixelSeedsRelaxed = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltL3SingleTauMETPixelSeeds = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltL3SingleTauMETPixelSeedsRelaxed = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltL3TauHighPtPixelSeeds = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltPFJetPixelSeeds = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()

hltBLifetimeRegionalPixelSeedGenerator = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltBLifetimeRegionalPixelSeedGeneratorSingleTop = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltBLifetimeRegionalPixelSeedGeneratorEleJetSingleTop = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltBLifetimeRegionalPixelSeedGeneratorIsoEleJetSingleTop = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltBLifetimeRegionalPixelSeedGeneratorRA2b = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltBLifetimeRegionalPixelSeedGeneratorRAzr = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltBLifetimeRegionalPixelSeedGeneratorHbb = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltBLifetimeRegionalPixel3DSeedGeneratorJet30Hbb = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltBLifetimeRegionalPixel3D1stTrkSeedGeneratorJet20Hbb = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltBLifetimeBTagIP3D1stTrkRegionalPixelSeedGeneratorJet20Hbb = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltBLifetimeDiBTagIP3D1stTrkRegionalPixelSeedGeneratorJet20Hbb = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltBLifetimeRegionalPixelSeedGeneratorbbPhi = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltBLifetimeRegionalPixelSeedGeneratorGammaB = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()

hltMumuPixelSeedFromL2Candidate = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltMumukPixelSeedFromL2Candidate = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltEcalRegionalPi0FEDs = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltEcalRegionalPi0Digis = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltEcalRegionalPi0WeightUncalibRecHit = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltEcalRegionalPi0RecHitTmp = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltFEDSelector = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
#hltEcalRegionalPi0RecHit = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
#
# Tracker = Pixel and Strip
#
from FastSimulation.HighLevelTrigger.RecoLocalTracker_cff import *
#
# Calorimeter = Ecal and Hcal
#
from FastSimulation.HighLevelTrigger.RecoLocalCalo_cff import *
from FastSimulation.HighLevelTrigger.EcalRegionalReco_cff import *
# Specific reconstruction sequences for FastSimulation
from FastSimulation.HighLevelTrigger.HLTFastReco_cff import *
HLTDoLocalPixelSequence           = cms.Sequence(pixeltrackerlocalreco)
hltSiPixelDigis = cms.Sequence(pixeltrackerlocalreco)
hltSiPixelClusters = cms.Sequence(pixeltrackerlocalreco)
hltSiPixelRecHits = cms.Sequence(pixeltrackerlocalreco)
HLTDoLocalPixelSequenceReg        = cms.Sequence(pixeltrackerlocalreco)
HLTDoLocalPixelSequenceRegL2Tau   = cms.Sequence(pixeltrackerlocalreco)
HLTDoLocalPixelSequenceRegForBTag = cms.Sequence(pixeltrackerlocalreco)
HLTDoLocalStripSequence           = cms.Sequence(striptrackerlocalreco)
HLTDoLocalStripSequenceReg        = cms.Sequence(striptrackerlocalreco)
HLTDoLocalStripSequenceRegForBTag = cms.Sequence(striptrackerlocalreco)

from FastSimulation.Tracking.HLTIterativeTracking_cff import *

hltMuonCSCDigis = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltMuonDTDigis = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltMuonRPCDigis = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltGtDigis = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()

