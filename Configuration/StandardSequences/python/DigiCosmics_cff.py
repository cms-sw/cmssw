import FWCore.ParameterSet.Config as cms

#                                                    
# Full-scale Digitization of the simulated hits      
# in all CMS subdets : Tracker, ECAL, HCAl, Muon's;  
# MixingModule (at least in zero-pileup mode) needs  
# to be included to make Digi's operational, since   
# it's required for ECAL/HCAL & Muon's                
# Defined in a separate fragment
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
#
# PPS Digis
# returns sequence "ctppsDigi"
from SimPPS.Configuration.SimPPS_cff import *

# include TrackingParticle Producer
# NOTA BENE: it MUST be run here at the moment, since it depends 
# of the availability of the CrossingFrame in the Event
#
from SimGeneral.Configuration.SimGeneral_cff import *

#from SimGeneral.MixingModule.mixNoPU_cfi import *

#Special parameterization for cosmics
#simSiPixelDigis.TofLowerCut = cms.double(18.5)
#simSiPixelDigis.TofUpperCut = cms.double(43.5)
#mix.digitizers.pixel.TofLowerCut = cms.double(18.5) moved to digitizersCosmics_cfi.py in the mixingmodule
#mix.digitizers.pixel.TofUpperCut = cms.double(43.5) moved to digitizersCosmics_cfi.py in the mixingmodule

#simSiStripDigis.CosmicDelayShift = cms.untracked.double(31)
#mix.digitizers.strip.CosmicDelayShift = cms.untracked.double(31) moved to digitizersCosmics_cfi.py in the mixingmodule

#simEcalUnsuppressedDigis.cosmicsPhase = cms.bool(True)
#simEcalUnsuppressedDigis.cosmicsShift = cms.double(1.)
#mix.digitizers.ecal.cosmicsPhase = cms.bool(True) moved to digitizersCosmics_cfi.py in the mixingmodule
#mix.digitizers.ecal.cosmicsShift = cms.double(1.) moved to digitizersCosmics_cfi.py in the mixingmodule

simEcalDigis.ebDccAdcToGeV = cms.double(0.00875)
simEcalDigis.srpBarrelLowInterestChannelZS = cms.double(0.0153125)

simHcalDigis.HBlevel = cms.int32(-10000)
simHcalDigis.HElevel = cms.int32(-10000)
simHcalDigis.HOlevel   = cms.int32(-10000)
simHcalDigis.HFlevel   = cms.int32(-10000)

doAllDigiTask = cms.Task(calDigiTask, muonDigiTask ,ctppsDigiTask)
pdigiTask = cms.Task(cms.TaskPlaceholder("randomEngineStateProducer"), cms.TaskPlaceholder("mix"), doAllDigiTask)

doAllDigi = cms.Sequence(doAllDigiTask)
pdigi = cms.Sequence(pdigiTask)
pdigi_valid = cms.Sequence(pdigiTask)

