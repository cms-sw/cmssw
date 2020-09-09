import FWCore.ParameterSet.Config as cms
from L1Trigger.CSCCommonTrigger.CSCCommonTrigger_cfi import *
from L1Trigger.CSCTriggerPrimitives.alctParams_cfi import *
from L1Trigger.CSCTriggerPrimitives.clctParams_cfi import *
from L1Trigger.CSCTriggerPrimitives.tmbParams_cfi import *
from L1Trigger.CSCTriggerPrimitives.auxiliaryParams_cfi import *
from L1Trigger.CSCTriggerPrimitives.cclutParams_cfi import *

cscTriggerPrimitiveDigis = cms.EDProducer(
    "CSCTriggerPrimitivesProducer",
    CSCCommonTrigger,
    cclutParams,

    # True: use parameters from this config
    # False: read parameters from DB using EventSetup mechanism
    debugParameters = cms.bool(False),

    # Name of digi producer modules
    CSCComparatorDigiProducer = cms.InputTag("simMuonCSCDigis","MuonCSCComparatorDigi"),
    CSCWireDigiProducer = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi"),
    GEMPadDigiProducer = cms.InputTag(""),
    GEMPadDigiClusterProducer = cms.InputTag(""),

    # If True, output collections will only be built for good chambers
    checkBadChambers = cms.bool(True),

    # Write out special trigger collections
    writeOutAllCLCTs = cms.bool(False),
    writeOutAllALCTs = cms.bool(False),
    savePreTriggers = cms.bool(False),

    alctPhase1 = alctPhase1,
    clctPhase1 = clctPhase1,
    tmbPhase1 = tmbPhase1,
    commonParam = commonParam,
    mpcParam  = mpcParamRun1,
)


## unganging in ME1/a
## no sorting/selecting in MPC
from Configuration.Eras.Modifier_run2_common_cff import run2_common
run2_common.toModify( cscTriggerPrimitiveDigis,
                      debugParameters = True,
                      checkBadChambers = False,
                      commonParam = dict(gangedME1a = False),
                      mpcParam = mpcParamRun2
)

## turn on upgrade CSC algorithm without GEMs
from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify( cscTriggerPrimitiveDigis,
                      commonParam = dict(isSLHC = True,
                                         runME11Up = cms.bool(True)),
                      ## ME1/1
                      alctPhase2ME11 = alctPhase2,
                      clctPhase2ME11 = clctPhase2,
                      tmbPhase2ME11 = tmbPhase2
)

## GEM-CSC ILT in ME1/1
from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toModify( cscTriggerPrimitiveDigis,
                   GEMPadDigiProducer = cms.InputTag("simMuonGEMPadDigis"),
                   GEMPadDigiClusterProducer = cms.InputTag("simMuonGEMPadDigiClusters"),
                   commonParam = dict(runME11ILT = cms.bool(True),
                                      useClusters = cms.bool(True)),
                   ## ME1/1
                   clctPhase2GE11 = clctPhase2.clone(clctNplanesHitPattern = 3),
                   tmbPhase2GE11 = tmbME11GEM,
                   copadParamGE11 = copadParamGE11
)

## GEM-CSC ILT in ME2/1
## upgrade algorithms in ME3/1 and ME4/1
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toModify( cscTriggerPrimitiveDigis,
                      commonParam = dict(runME21Up = cms.bool(True),
                                         runME21ILT = cms.bool(True),
                                         runME31Up = cms.bool(True),
                                         runME41Up = cms.bool(True)),
                      ## MEX/1 without GEMs
                      tmbPhase2MEX1 = tmbMEX1,
                      alctPhase2MEX1 = alctPhase1,
                      clctPhase2MEX1 = clctPhase2,
                      ## ME2/1 with GEM
                      alctPhase2GE21 = alctPhase1.clone(alctNplanesHitPattern = 3),
                      clctPhase2GE21 = clctPhase2.clone(clctNplanesHitPattern = 3),
                      tmbPhase2GE21 = tmbME21GEM,
                      copadParamGE21 = copadParamGE21,
)
