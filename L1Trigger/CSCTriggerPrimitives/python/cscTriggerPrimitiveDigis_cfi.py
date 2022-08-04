import FWCore.ParameterSet.Config as cms

## Import minBX and maxBX for the MPC
from L1Trigger.CSCTriggerPrimitives.CSCCommonTrigger_cfi import CSCCommonTrigger

## The directory params/ contains common psets, psets for the ALCT and CLCT
## processsors, for the TMBs, MPC and CCLUT, the GEM-CSC integrated local trigger,
## and the hadronic shower trigger.
from L1Trigger.CSCTriggerPrimitives.params.alctParams import alctPSets
from L1Trigger.CSCTriggerPrimitives.params.clctParams import clctPSets
from L1Trigger.CSCTriggerPrimitives.params.tmbParams import tmbPSets
from L1Trigger.CSCTriggerPrimitives.params.auxiliaryParams import auxPSets
from L1Trigger.CSCTriggerPrimitives.params.gemcscParams import gemcscPSets
from L1Trigger.CSCTriggerPrimitives.params.showerParams import showerPSet

cscTriggerPrimitiveDigis = cms.EDProducer(
    "CSCTriggerPrimitivesProducer",
    CSCCommonTrigger,
    ## pass all processor parameters sets
    ## deal with the customization in the CSCBaseboard class...
    alctPSets,
    clctPSets,
    tmbPSets,
    gemcscPSets,

    # True: use parameters from this config
    # False: read parameters from DB using EventSetup mechanism
    debugParameters = cms.bool(False),

    # Name of digi producer modules
    CSCComparatorDigiProducer = cms.InputTag("simMuonCSCDigis","MuonCSCComparatorDigi"),
    CSCWireDigiProducer = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi"),
    GEMPadDigiClusterProducer = cms.InputTag(""),

    # If True, output collections will only be built for good chambers
    checkBadChambers = cms.bool(True),

    # Anode-DAQ rate determined by pre-CLCTs
    keepCLCTPreTriggers = cms.bool(True),
    # Anode-DAQ rate determined by ALCTs
    keepALCTPreTriggers = cms.bool(False),
    # special triggers for showers in chambers
    keepShowers = cms.bool(False),

    commonParam = auxPSets.commonParam.clone(),
    mpcParam = auxPSets.mpcParamRun1.clone(),
    showerParam = showerPSet.clone()
)


## unganging in ME1/a
## no sorting/selecting in MPC
from Configuration.Eras.Modifier_run2_common_cff import run2_common
run2_common.toModify( cscTriggerPrimitiveDigis,
                      debugParameters = True,
                      checkBadChambers = False,
                      commonParam = dict(gangedME1a = False),
                      mpcParam = auxPSets.mpcParamRun2.clone()
)

## turn on upgrade CSC algorithm without GEMs
## originally intended for Phase-2, but moved up to Run-3
from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify( cscTriggerPrimitiveDigis,
                      keepShowers = True,
                      commonParam = dict(run3 = True,
                                         runCCLUT_OTMB = True,
                                         runPhase2 = True,
                                         runME11Up = True,
                                         runME21Up = True,
                                         runME31Up = True,
                                         runME41Up = True)
)

## GEM-CSC integrated local trigger in ME1/1
from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toModify( cscTriggerPrimitiveDigis,
                   GEMPadDigiClusterProducer = cms.InputTag("simMuonGEMPadDigiClusters"),
                   commonParam = dict(runME11ILT = True)
)

## GEM-CSC integrated local trigger in ME2/1
## enable the Phase-2 ALCT processors
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toModify( cscTriggerPrimitiveDigis,
                      commonParam = dict(runME21ILT = True,
                                         enableAlctPhase2 = True)
)
