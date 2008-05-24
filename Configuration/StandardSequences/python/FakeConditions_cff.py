import FWCore.ParameterSet.Config as cms

# Tracker alignment and calib
# from a single entry point
from CalibTracker.Configuration.Tracker_FakeConditions_cff import *
#
# Muon alignment
#
from CalibMuon.Configuration.Muon_FakeAlignment_cff import *
#
#
# DT calib
from CalibMuon.Configuration.DT_FakeConditions_cff import *
#
# RPC noise
from CalibMuon.Configuration.RPC_FakeConditions_cff import *
#
# CSC Calib
#
# DBCOnditions use the new DB objects (linearized vectors)
from CalibMuon.Configuration.CSC_FakeDBConditions_cff import *
#
# HCAL Fake Conditions 
#
from CalibCalorimetry.Configuration.Hcal_FakeConditions_cff import *
#
# ECAL Fake Conditions
#
from CalibCalorimetry.Configuration.Ecal_FakeConditions_cff import *
#
# Btag conditions
#
from RecoBTag.Configuration.RecoBTag_FakeConditions_cff import *
# Btau
#
from RecoBTau.Configuration.RecoBTau_FakeConditions_cff import *
# BeamSpot Conditions
from RecoVertex.BeamSpotProducer.BeamSpotFakeConditionsEarlyCollision_cff import *
# Cabling maps
from EventFilter.DTRawToDigi.DTSQLiteCabling_cfi import *
from EventFilter.RPCRawToDigi.RPCSQLiteCabling_cfi import *
from EventFilter.CSCRawToDigi.cscSQLiteCablingPack_cff import *
from EventFilter.CSCRawToDigi.cscSQLiteCablingUnpck_cff import *


