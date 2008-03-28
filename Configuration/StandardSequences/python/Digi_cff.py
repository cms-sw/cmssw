import FWCore.ParameterSet.Config as cms

#
# Full-scale Digitization of the simulated hits 
# in all CMS subdets : Tracker, ECAL, HCAl, Muon's; 
# MixingModule (at least in zero-pileup mode) needs
# to be included to make Digi's operational, since 
# it's required for ECA/HCAL & Muon's
#
# Tracker Digis (Pixel + SiStrips)
# returns sequence "trDigi"
#
from SimTracker.Configuration.SimTracker_cff import *
# Calorimetry Digis (Ecal + Hcal) - * unsuppressed *
# returns sequence "calDigi"
from SimCalorimetry.Configuration.SimCalorimetry_cff import *
# Muon Digis (CSC + DT + RPC)
# returns sequence "muonDigi"
#
from SimMuon.Configuration.SimMuon_cff import *
doAllDigi = cms.Sequence(trDigi+calDigi+muonDigi)

