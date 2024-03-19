import FWCore.ParameterSet.Config as cms

from L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi import cscTriggerPrimitiveDigis

# Taken from L1Trigger.L1TMuon.simDigis_cff
simCscTriggerPrimitiveDigisForEMTF = cscTriggerPrimitiveDigis.clone(
    CSCComparatorDigiProducer = 'simMuonCSCDigis:MuonCSCComparatorDigi',
    CSCWireDigiProducer = 'simMuonCSCDigis:MuonCSCWireDigi'
)

# Taken from L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi
from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toModify(simCscTriggerPrimitiveDigisForEMTF,
                  commonParam = dict(runPhase2 = cms.bool(True),
                                     runME11Up = cms.bool(True),
                                     runME11ILT = cms.bool(False),  # was: True
                                     GEMPadDigiClusterProducer = cms.InputTag(""),
                                     enableAlctPhase2 = cms.bool(False)))  # was: True

from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toModify(simCscTriggerPrimitiveDigisForEMTF,
                     commonParam = dict(runME21Up = cms.bool(True),
                                        runME21ILT = cms.bool(False),  # was: True
                                        runME31Up = cms.bool(True),
                                        runME41Up = cms.bool(True),
                                        enableAlctPhase2 = cms.bool(False)))  # was: True

# Allow CSCs to have hits in multiple bxs - (Needs to be fixed on their end eventually)
phase2_muon.toModify(simCscTriggerPrimitiveDigisForEMTF.tmbPhase1, tmbReadoutEarliest2 = False)
phase2_muon.toModify(simCscTriggerPrimitiveDigisForEMTF.tmbPhase2, tmbReadoutEarliest2 = False)
