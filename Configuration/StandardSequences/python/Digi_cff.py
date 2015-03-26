import FWCore.ParameterSet.Config as cms

#                                                    
# Full-scale Digitization of the simulated hits      
# in all CMS subdets : Tracker, ECAL, HCAl, Muon's;  
# MixingModule (at least in zero-pileup mode) needs  
# to be included to make Digi's operational, since   
# it's required for ECAL/HCAL & Muon's                
# Defined in a separate fragment
#                                                    
# Tracker Digis (Pixel + SiStrips) are now made in the mixing
# module, so the old "trDigi" sequence has been taken out.
#

# Calorimetry Digis (Ecal + Hcal) - * unsuppressed *
# returns sequence "calDigi"
from SimCalorimetry.Configuration.SimCalorimetry_cff import *
# Muon Digis (CSC + DT + RPC)
# returns sequence "muonDigi"
#
from SimMuon.Configuration.SimMuon_cff import *
#
# TrackingParticle Producer is now part of the mixing module, so
# it is no longer run here.
#
from SimGeneral.Configuration.SimGeneral_cff import *

# add updating the GEN information by default
from Configuration.StandardSequences.Generator_cff import *

doAllDigi = cms.Sequence(calDigi+muonDigi)
pdigi = cms.Sequence(fixGenInfo*cms.SequencePlaceholder("randomEngineStateProducer")*cms.SequencePlaceholder("mix")*doAllDigi*addPileupInfo)
pdigi_valid = cms.Sequence(pdigi)
pdigi_nogen=cms.Sequence(cms.SequencePlaceholder("randomEngineStateProducer")*cms.SequencePlaceholder("mix")*doAllDigi*addPileupInfo)
pdigi_valid_nogen=cms.Sequence(pdigi_nogen)
