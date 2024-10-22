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
# PPS Digis
# returns sequence "ctppsDigi"
from SimPPS.Configuration.SimPPS_cff import *
#
# TrackingParticle Producer is now part of the mixing module, so
# it is no longer run here.
#
from SimGeneral.Configuration.SimGeneral_cff import *

from SimTracker.Configuration.SimTrackerLinks_cff import *

# add updating the GEN information by default
from Configuration.StandardSequences.Generator_cff import *
from GeneratorInterface.Core.generatorSmeared_cfi import *

doAllDigiTask = cms.Task(generatorSmeared, calDigiTask, muonDigiTask, ctppsDigiTask)
from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
# premixing stage2 runs muon digis after PreMixingModule (configured in DataMixerPreMix_cff)
premix_stage2.toReplaceWith(doAllDigiTask, doAllDigiTask.copyAndExclude([muonDigiTask]))

pdigiTask_nogen = cms.Task(generatorSmeared, cms.TaskPlaceholder("randomEngineStateProducer"), cms.TaskPlaceholder("mix"), doAllDigiTask, addPileupInfo)
# premixing stage2 runs addPileupInfo after PreMixingModule (configured in DataMixerPreMix_cff)
premix_stage2.toReplaceWith(pdigiTask_nogen, pdigiTask_nogen.copyAndExclude([addPileupInfo]))

pdigiTask = cms.Task(pdigiTask_nogen, fixGenInfoTask, tpPruningTask)

doAllDigi = cms.Sequence(doAllDigiTask)
pdigi = cms.Sequence(pdigiTask)
pdigi_valid = cms.Sequence(pdigiTask)
pdigi_nogen=cms.Sequence(pdigiTask_nogen)
pdigi_valid_nogen=cms.Sequence(pdigiTask_nogen)

from GeneratorInterface.HiGenCommon.HeavyIon_cff import *
pdigiTask_hi = cms.Task(pdigiTask, heavyIon)
pdigiTask_hi_nogen = cms.Task(pdigiTask_nogen, genJetMETTask, heavyIon)
pdigi_hi=cms.Sequence(pdigiTask_hi)
pdigi_hi_nogen=cms.Sequence(pdigiTask_hi_nogen)

# define genPUProtons as an EDProducer only when not in premixing stage2 job
# in premixing stage2 genPUProtons is an EDAlias, defined in aliases_PreMix_cfi
def _premixStage2GenPUProtons(process):
    process.load("SimGeneral.PileupInformation.genPUProtons_cfi")
    process.pdigiTask_nogen.add(process.genPUProtons)
modifyDigi_premixStage2GenPUProtons = (~premix_stage2).makeProcessModifier(_premixStage2GenPUProtons)

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
