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

#for backward compatibility of HepMCProduct
from GeneratorInterface.Core.generatorSmeared_cfi import *

# Calorimetry Digis (Ecal + Hcal) - * unsuppressed *
# returns sequence "calDigi"
from SimCalorimetry.Configuration.SimCalorimetry_cff import *
# Muon Digis (CSC + DT + RPC)
# returns sequence "muonDigi"
#
from SimMuon.Configuration.SimMuon_cff import *

# add updating the GEN information by default
from Configuration.StandardSequences.Generator_cff import *

doAllDigi = cms.Sequence(generatorSmeared*calDigi+muonDigi)
pdigi = cms.Sequence(generatorSmeared*fixGenInfo*cms.SequencePlaceholder("randomEngineStateProducer")*cms.SequencePlaceholder("mix")*doAllDigi)
pdigi_valid = cms.Sequence(pdigi)
# for PreMixing, to first approximation, allow noise in Muon system

# remove unnecessary modules from 'pdigi' sequence - run after DataMixing
# standard mixing module now makes unsuppressed digis for calorimeter
pdigi.remove(simEcalTriggerPrimitiveDigis)
pdigi.remove(simEcalDigis)  # does zero suppression
pdigi.remove(simEcalPreshowerDigis)  # does zero suppression
pdigi.remove(simHcalDigis)
pdigi.remove(simHcalTTPDigis)

