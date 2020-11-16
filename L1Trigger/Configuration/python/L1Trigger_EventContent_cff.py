import FWCore.ParameterSet.Config as cms

# RAW content
L1TriggerRAW = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep  FEDRawDataCollection_rawDataCollector_*_*',
        'keep  FEDRawDataCollection_source_*_*')
)


# RAWDEBUG content
L1TriggerRAWDEBUG = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep  FEDRawDataCollection_rawDataCollector_*_*',
        'keep  FEDRawDataCollection_source_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_l1GtRecord_*_*',
        'keep *_l1GtTriggerMenuLite_*_*',
        'keep *_conditionsInEdm_*_*',
        'keep *_l1extraParticles_*_*')
)

# RECO content
L1TriggerRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_l1GtRecord_*_*',
        'keep *_l1GtTriggerMenuLite_*_*',
        'keep *_conditionsInEdm_*_*',
        'keep *_l1extraParticles_*_*',
        'keep *_l1L1GtObjectMap_*_*',
        'keep L1MuGMTReadoutCollection_gtDigis_*_*',
        'keep L1GctEmCand*_gctDigis_*_*',
        'keep L1GctJetCand*_gctDigis_*_*',
        'keep L1GctEtHad*_gctDigis_*_*',
        'keep L1GctEtMiss*_gctDigis_*_*',
        'keep L1GctEtTotal*_gctDigis_*_*',
        'keep L1GctHtMiss*_gctDigis_*_*',
        'keep L1GctJetCounts*_gctDigis_*_*',
        'keep L1GctHFRingEtSums*_gctDigis_*_*',
        'keep L1GctHFBitCounts*_gctDigis_*_*',
        'keep LumiDetails_lumiProducer_*_*',
        'keep LumiSummary_lumiProducer_*_*')
)


# AOD content
L1TriggerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_l1GtRecord_*_*',
        'keep *_l1GtTriggerMenuLite_*_*',
        'keep *_conditionsInEdm_*_*',
        'keep *_l1extraParticles_*_*',
        'keep *_l1L1GtObjectMap_*_*',
        'keep LumiSummary_lumiProducer_*_*')
)


L1TriggerFEVTDEBUG = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_simCscTriggerPrimitiveDigis_*_*',
        'keep *_simDtTriggerPrimitiveDigis_*_*',
        'keep *_simRpcTriggerDigis_*_*',
        'keep *_simRctDigis_*_*',
        'keep *_simCsctfDigis_*_*',
        'keep *_simCsctfTrackDigis_*_*',
        'keep *_simDttfDigis_*_*',
        'keep *_simGctDigis_*_*',
        'keep *_simCaloStage1Digis_*_*',
        'keep *_simCaloStage1FinalDigis_*_*',
        'keep *_simCaloStage2Layer1Digis_*_*',
        'keep *_simCaloStage2Digis_*_*',
        'keep *_simGmtDigis_*_*',
        "keep *_simBmtfDigis_*_*",
        "keep *_simKBmtfDigis_*_*",
        "keep *_simOmtfDigis_*_*",
        "keep *_simEmtfDigis_*_*",
        "keep *_simGmtStage2Digis_*_*",
        'keep *_simGtDigis_*_*',
        "keep *_simGtStage2Digis_*_*",
        'keep *_cscTriggerPrimitiveDigis_*_*',
        'keep *_dtTriggerPrimitiveDigis_*_*',
        'keep *_rpcTriggerDigis_*_*',
        'keep *_rctDigis_*_*',
        'keep *_csctfDigis_*_*',
        'keep *_csctfTrackDigis_*_*',
        'keep *_dttfDigis_*_*',
        'keep *_gctDigis_*_*',
        'keep *_gmtDigis_*_*',
        'keep *_gtDigis_*_*',
        'keep *_gtEvmDigis_*_*',
        'keep *_l1GtRecord_*_*',
        'keep *_l1GtTriggerMenuLite_*_*',
        'keep *_conditionsInEdm_*_*',
        'keep *_l1extraParticles_*_*',
        'keep *_l1L1GtObjectMap_*_*',
        'keep LumiDetails_lumiProducer_*_*',
        'keep LumiSummary_lumiProducer_*_*')
)


def _appendStage2Digis(obj):
    l1Stage2Digis = [
        'keep *_gtStage2Digis_*_*',
        'keep *_gmtStage2Digis_*_*',
        'keep *_caloStage2Digis_*_*',
        ]
    obj.outputCommands += l1Stage2Digis

# adding them to all places where we had l1extraParticles
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
stage2L1Trigger.toModify(L1TriggerRAWDEBUG, func=_appendStage2Digis)
stage2L1Trigger.toModify(L1TriggerRECO, func=_appendStage2Digis)
stage2L1Trigger.toModify(L1TriggerAOD, func=_appendStage2Digis)
stage2L1Trigger.toModify(L1TriggerFEVTDEBUG, func=_appendStage2Digis)

# adding HGCal L1 trigger digis
def _appendHGCalDigis(obj):
    l1HGCalDigis = [
        'keep l1tHGCalTriggerCellBXVector_hgcalVFEProducer_*_*',
        'keep l1tHGCalTriggerCellBXVector_hgcalConcentratorProducer_*_*',
        'keep l1tHGCalTowerBXVector_hgcalTowerProducer_*_*',
        'keep l1tHGCalClusterBXVector_hgcalBackEndLayer1Producer_*_*',
        'keep l1tHGCalMulticlusterBXVector_hgcalBackEndLayer2Producer_*_*'
        ]
    obj.outputCommands += l1HGCalDigis

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toModify(L1TriggerFEVTDEBUG, func=_appendHGCalDigis)

# adding GEM trigger primitives
def _appendGEMDigis(obj):
    l1GEMDigis = [
        'keep *_simMuonGEMPadDigis_*_*',
        'keep *_simMuonGEMPadDigiClusters_*_*',
        ]
    obj.outputCommands += l1GEMDigis

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toModify(L1TriggerFEVTDEBUG, func=_appendGEMDigis)

# adding ME0 trigger primitives
def _appendME0Digis(obj):
    l1ME0Digis = [
        'keep *_simMuonME0PadDigis__*',
        'keep *_me0TriggerDigis__*',
        'keep *_simMuonME0PseudoReDigisCoarse__*',
        'keep *_me0RecHitsCoarse__*',
        'keep *_me0TriggerPseudoDigis__*',
        'keep *_me0TriggerConvertedPseudoDigis__*',
        ]
    obj.outputCommands += l1ME0Digis

from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toModify(L1TriggerFEVTDEBUG, func=_appendME0Digis)

# adding phase2 trigger
def _appendPhase2Digis(obj):
    l1Phase2Digis = [
        "keep *_simKBmtfDigis_*_*",
        'keep *_hgcalVFEProducerhgcalConcentratorProducer_*_*',
        'keep *_hgcalBackEndLayer1Producer_*_*',
        'keep *_hgcalBackEndLayer2Producer_*_*',
        'keep *_hgcalTowerMapProducer_*_*',
        'keep *_hgcalTowerProducer_*_*',
        'keep *_L1EGammaClusterEmuProducer_*_*',
        'keep *_l1EGammaEEProducer_*_*',
        'keep *_L1TkPrimaryVertex_*_*',
        'keep *_L1TkElectronsCrystal_*_*',
        'keep *_L1TkElectronsLooseCrystal_*_*',
        'keep *_L1TkElectronsEllipticMatchCrystal_*_*',
        'keep *_L1TkIsoElectronsCrystal_*_*',
        'keep *_L1TkPhotonsCrystal_*_*',
        'keep *_L1TkElectronsHGC_*_*',
        'keep *_L1TkElectronsEllipticMatchHGC_*_*',
        'keep *_L1TkIsoElectronsHGC_*_*',
        'keep *_L1TkPhotonsHGC_*_*',
        'keep *_L1TkMuons_*_*',
        'keep *_pfClustersFromL1EGClusters_*_*',
        'keep *_pfClustersFromCombinedCaloHCal_*_*',
        'keep *_pfClustersFromCombinedCaloHF_*_*',
        'keep *_pfClustersFromHGC3DClusters_*_*',
        'keep *_pfTracksFromL1TracksBarrel_*_*',
        'keep *_l1pfProducerBarrel_*_*',
        'keep *_pfTracksFromL1TracksHGCal_*_*',
        'keep *_l1pfProducerHGCal_*_*',
        'keep *_l1pfProducerHGCalNoTK_*_*',
        'keep *_l1pfProducerHF_*_*',
        'keep *_l1pfCandidates_*_*',
        'keep *_ak4PFL1Calo_*_*',
        'keep *_ak4PFL1PF_*_*',
        'keep *_ak4PFL1Puppi_*_*',
        'keep *_ak4PFL1CaloCorrected_*_*',
        'keep *_ak4PFL1PFCorrected_*_*',
        'keep *_ak4PFL1PuppiCorrected_*_*',
        'keep *_l1PFMetCalo_*_*',
        'keep *_l1PFMetPF_*_*',
        'keep *_l1PFMetPuppi_*_*',
        'keep *_TTStubsFromPhase2TrackerDigis_*_*',
        'keep *_TTClustersFromPhase2TrackerDigis_*_*',
        'keep *_TTTracksFromExtendedTrackletEmulation_*_*',
        'keep *_TTTracksFromTrackletEmulation_*_*',
        ]
    obj.outputCommands += l1Phase2Digis

from Configuration.Eras.Modifier_phase2_trigger_cff import phase2_trigger
phase2_muon.toModify(L1TriggerFEVTDEBUG, func=_appendPhase2Digis)
