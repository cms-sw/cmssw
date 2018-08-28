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
from GeneratorInterface.Core.generatorSmeared_cfi import *
from SimGeneral.PileupInformation.genPUProtons_cfi import *

doAllDigi = cms.Sequence(generatorSmeared*calDigi+muonDigi)
pdigi = cms.Sequence(generatorSmeared*fixGenInfo*cms.SequencePlaceholder("randomEngineStateProducer")*cms.SequencePlaceholder("mix")*doAllDigi*addPileupInfo*genPUProtons)
pdigi_valid = cms.Sequence(pdigi)
pdigi_nogen=cms.Sequence(generatorSmeared*cms.SequencePlaceholder("randomEngineStateProducer")*cms.SequencePlaceholder("mix")*doAllDigi*addPileupInfo*genPUProtons)
pdigi_valid_nogen=cms.Sequence(pdigi_nogen)

from GeneratorInterface.HiGenCommon.HeavyIon_cff import *
pdigi_hi=cms.Sequence(pdigi+heavyIon)
pdigi_hi_nogen=cms.Sequence(pdigi_nogen+genJetMET+heavyIon)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
def _fastSimDigis(process):
    import FastSimulation.Configuration.DigiAliases_cff as DigiAliases

    # pretend these digis have been through digi2raw and raw2digi, by using the approprate aliases
    # use an alias to make the mixed track collection available under the usual label
    from FastSimulation.Configuration.DigiAliases_cff import loadDigiAliases
    loadDigiAliases(process)
# no need for the aliases for premixing stage1
modifyDigi_fastSimDigis = (fastSim & ~premix_stage1).makeProcessModifier(_fastSimDigis)

#phase 2 common mods
def _modifyEnableHcalHardcode( theProcess ):
    from CalibCalorimetry.HcalPlugins.Hcal_Conditions_forGlobalTag_cff import hcal_db_producer as _hcal_db_producer, es_hardcode as _es_hardcode, es_prefer_hcalHardcode as _es_prefer_hcalHardcode
    theProcess.hcal_db_producer = _hcal_db_producer
    theProcess.es_hardcode = _es_hardcode
    theProcess.es_prefer_hcalHardcode = _es_prefer_hcalHardcode    

from Configuration.Eras.Modifier_hcalHardcodeConditions_cff import hcalHardcodeConditions
modifyEnableHcalHardcode_ = hcalHardcodeConditions.makeProcessModifier( _modifyEnableHcalHardcode )


