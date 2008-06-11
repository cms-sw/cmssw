import FWCore.ParameterSet.Config as cms

# RecoMuon flux ##########################################################
# L2 seeds from L1 input
# module hltL2MuonSeeds = L2MuonSeeds from "RecoMuon/L2MuonSeedGenerator/data/L2MuonSeeds.cfi"
# replace hltL2MuonSeeds.GMTReadoutCollection = l1ParamMuons
# replace hltL2MuonSeeds.InputObjects = l1ParamMuons
# L3 regional reconstruction
from FastSimulation.Muons.L3Muons_cff import *
import FastSimulation.Muons.L3Muons_cfi
hltL3Muons = FastSimulation.Muons.L3Muons_cfi.L3Muons.clone()
# L3 regional seeding, candidating, tracking
from FastSimulation.Muons.TSGFromL2_cfi import *
from FastSimulation.Muons.TrackCandidateFromL2_cfi import *
from FastSimulation.Muons.HLTL3TkMuons_cfi import *
hltL3TrackCandidateFromL2 = cms.Sequence(hltL3CandidateFromL2+hltL3TkMuons)


