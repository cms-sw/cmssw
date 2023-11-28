import FWCore.ParameterSet.Config as cms

MicroEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_slimmedPhotons_*_*',
        'keep *_slimmedOOTPhotons_*_*',
        'keep *_slimmedElectrons_*_*',
        'keep *_slimmedMuons_*_*',
        'keep recoTrackExtras_slimmedMuonTrackExtras_*_*',
        'keep TrackingRecHitsOwned_slimmedMuonTrackExtras_*_*',
        'keep SiPixelClusteredmNewDetSetVector_slimmedMuonTrackExtras_*_*',
        'keep SiStripClusteredmNewDetSetVector_slimmedMuonTrackExtras_*_*',
        'keep *_slimmedTaus_*_*',
        'keep *_slimmedTausBoosted_*_*',
        'keep *_slimmedCaloJets_*_*',
        'keep *_slimmedJPTJets_*_*',
        'keep *_slimmedJets_*_*',
        # keep slimmedJets TagInfos, currently only PixelClusterTagInfo
        'keep recoBaseTagInfosOwned_slimmedJets_*_*',
        'keep *_slimmedJetsAK8_*_*',
        # drop content created by MINIAOD DeepDoubleB production
        'drop recoBaseTagInfosOwned_slimmedJetsAK8_*_*',
        'keep *_slimmedJetsPuppi_*_*',
        'keep *_slimmedMETs_*_*',
        'keep *_slimmedMETsNoHF_*_*',
        'keep *_slimmedMETsPuppi_*_*',
        'keep *_slimmedSecondaryVertices_*_*',
        'keep *_slimmedLambdaVertices_*_*',
        'keep *_slimmedKshortVertices_*_*',
        'keep *_slimmedJetsAK8PFPuppiSoftDropPacked_SubJets_*',

        'keep recoPhotonCores_reducedEgamma_*_*',
        'keep recoGsfElectronCores_reducedEgamma_*_*',
        'keep recoConversions_reducedEgamma_*_*',
        'keep recoSuperClusters_reducedEgamma_*_*',
        'keep recoCaloClusters_reducedEgamma_*_*',
        'keep EcalRecHitsSorted_reducedEgamma_*_*',
        'keep recoGsfTracks_reducedEgamma_*_*',
        'keep HBHERecHitsSorted_reducedEgamma_*_*',
        'keep *_slimmedHcalRecHits_*_*',
        'drop *_*_caloTowers_*',
        'drop *_*_pfCandidates_*',
        'drop *_*_genJets_*',
        'keep *_offlineBeamSpot_*_*',
        'keep *_offlineSlimmedPrimaryVertices_*_*',
        'keep *_offlineSlimmedPrimaryVerticesWithBS_*_*',
        'keep patPackedCandidates_packedPFCandidates_*_*',
        'keep *_isolatedTracks_*_*',
        # low energy conversions for BPH
        'keep *_oniaPhotonCandidates_*_*',

        'keep *_bunchSpacingProducer_*_*',

        'keep double_fixedGridRhoAll__*',
        'keep double_fixedGridRhoFastjetAll__*',
        'keep double_fixedGridRhoFastjetAllTmp__*',
        'keep double_fixedGridRhoFastjetAllCalo__*',
        'keep double_fixedGridRhoFastjetCentral_*_*',
        'keep double_fixedGridRhoFastjetCentralCalo__*',
        'keep double_fixedGridRhoFastjetCentralChargedPileUp__*',
        'keep double_fixedGridRhoFastjetCentralNeutral__*',

        'keep *_slimmedPatTrigger_*_*',
        'keep patPackedTriggerPrescales_patTrigger__*',
        'keep patPackedTriggerPrescales_patTrigger_l1max_*',
        'keep patPackedTriggerPrescales_patTrigger_l1min_*',
        # old L1 trigger
        'keep *_l1extraParticles_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        # stage 2 L1 trigger
        'keep GlobalExtBlkBXVector_simGtExtUnprefireable_*_*',
        'keep *_gtStage2Digis__*',
        'keep *_gmtStage2Digis_Muon_*',
        'keep *_caloStage2Digis_Jet_*',
        'keep *_caloStage2Digis_Tau_*',
        'keep *_caloStage2Digis_EGamma_*',
        'keep *_caloStage2Digis_EtSum_*',
        # HLT
        'keep *_TriggerResults_*_HLT',
        'keep *_TriggerResults_*_*', # for MET filters (a catch all for the moment, but ideally it should be only the current process)
        'keep patPackedCandidates_lostTracks_*_*',
        'keep HcalNoiseSummary_hcalnoise__*',
        'keep recoCSCHaloData_CSCHaloData_*_*',
        'keep recoBeamHaloSummary_BeamHaloSummary_*_*',
        # Lumi
        'keep LumiScalerss_scalersRawToDigi_*_*',
        # CTPPS
        'keep CTPPSLocalTrackLites_ctppsLocalTrackLiteProducer_*_*',
        'keep recoForwardProtons_ctppsProtons_*_*',
	# displacedStandAlone muon collection for EXO
	'keep recoTracks_displacedStandAloneMuons__*',
        'keep recoTracks_displacedGlobalMuons__*',
        'keep recoTracks_displacedTracks__*',
        # L1 prefiring weights
        'keep *_prefiringweight_*_*',
        # patLowPtElectrons
        'keep *_slimmedLowPtElectrons_*_*',
        'keep *_gsfTracksOpenConversions_*_*',
        # patDisplacedMuons
        'keep *_slimmedDisplacedMuons_*_*',
        'keep recoTrackExtras_slimmedDisplacedMuonTrackExtras_*_*',
        'keep TrackingRecHitsOwned_slimmedDisplacedMuonTrackExtras_*_*',
        'keep SiPixelClusteredmNewDetSetVector_slimmedDisplacedMuonTrackExtras_*_*',
        'keep SiStripClusteredmNewDetSetVector_slimmedDisplacedMuonTrackExtras_*_*',
    )
)

MicroEventContentGEN = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep patPackedGenParticles_packedGenParticles_*_*',
        'keep recoGenParticles_prunedGenParticles_*_*',
        'keep *_packedPFCandidateToGenAssociation_*_*',
        'keep *_lostTracksToGenAssociation_*_*',
        'keep LHEEventProduct_*_*_*',
        'keep GenFilterInfo_*_*_*',
        'keep GenLumiInfoHeader_generator_*_*',
        'keep GenLumiInfoProduct_*_*_*',
        'keep GenEventInfoProduct_generator_*_*',
        'keep recoGenParticles_genPUProtons_*_*',
        'keep *_slimmedGenJetsFlavourInfos_*_*',
        'keep *_slimmedGenJets__*',
        'keep *_slimmedGenJetsAK8__*',
        'keep *_slimmedGenJetsAK8SoftDropSubJets__*',
        'keep *_genMetTrue_*_*',
        # RUN
        'keep LHERunInfoProduct_*_*_*',
        'keep GenRunInfoProduct_*_*_*',
        'keep *_genParticles_xyz0_*',
        'keep *_genParticles_t0_*',
    )
)

# --- Only for 2018 data & MC
_run2_HCAL_2018_extraCommands = ["keep *_packedPFCandidates_hcalDepthEnergyFractions_*"]
from Configuration.Eras.Modifier_run2_HCAL_2018_cff import run2_HCAL_2018
run2_HCAL_2018.toModify(MicroEventContent, outputCommands = MicroEventContent.outputCommands + _run2_HCAL_2018_extraCommands)

_run3_common_extraCommands = ["drop *_packedPFCandidates_hcalDepthEnergyFractions_*"]
from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(MicroEventContent, outputCommands = MicroEventContent.outputCommands + _run3_common_extraCommands)
# ---

_pp_on_AA_extraCommands = [
    'keep patPackedCandidates_hiPixelTracks_*_*',
    'keep patPackedCandidates_packedPFCandidatesRemoved_*_*',
    'keep *_packedCandidateMuonID_*_*',
    'keep *_slimmedJets_pfCandidates_*',
    'keep floatedmValueMap_packedPFCandidateTrackChi2_*_*',
    'keep floatedmValueMap_lostTrackChi2_*_*',
    'keep recoCentrality_hiCentrality_*_*',
    'keep int_centralityBin_*_*',
    'keep recoHFFilterInfo_hiHFfilters_*_*',
    'keep recoClusterCompatibility_hiClusterCompatibility_*_*',
    'keep *_offlineSlimmedPrimaryVerticesRecovery_*_*',
    'keep *_hiEvtPlane_*_*',
    'keep *_hiEvtPlaneFlat_*_*',
    'keep QIE10DataFrameHcalDataFrameContainer_hcalDigis_ZDC_*',
]

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(MicroEventContent, outputCommands = MicroEventContent.outputCommands + _pp_on_AA_extraCommands)

_zdc_extraCommands = ['keep QIE10DataFrameHcalDataFrameContainer_hcalDigis_ZDC_*']
from Configuration.ProcessModifiers.storeZDCDigis_cff import storeZDCDigis
storeZDCDigis.toModify(MicroEventContent, outputCommands = MicroEventContent.outputCommands + _zdc_extraCommands)

MicroEventContentMC = cms.PSet(
    outputCommands = cms.untracked.vstring(MicroEventContent.outputCommands)
)
MicroEventContentMC.outputCommands += MicroEventContentGEN.outputCommands
MicroEventContentMC.outputCommands += [
                                        'keep PileupSummaryInfos_slimmedAddPileupInfo_*_*',
                                        # RUN
                                        'keep L1GtTriggerMenuLite_l1GtTriggerMenuLite__*'
                                      ]
_pp_on_AA_MC_extraCommands = ['keep *_packedGenParticlesSignal_*_*','keep edmGenHIEvent_heavyIon_*_*']
pp_on_AA.toModify(MicroEventContentMC, outputCommands = MicroEventContentMC.outputCommands + _pp_on_AA_MC_extraCommands)

from Configuration.Eras.Modifier_strips_vfp30_2016_cff import strips_vfp30_2016
strips_vfp30_2016.toModify(MicroEventContentMC, outputCommands = MicroEventContentMC.outputCommands + [
    'keep *_simAPVsaturation_SimulatedAPVDynamicGain_*'
])

MiniAODOverrideBranchesSplitLevel = cms.untracked.VPSet( [
cms.untracked.PSet(branch = cms.untracked.string("patPackedCandidates_packedPFCandidates__*"),splitLevel=cms.untracked.int32(99)),
cms.untracked.PSet(branch = cms.untracked.string("recoGenParticles_prunedGenParticles__*"),splitLevel=cms.untracked.int32(99)),
cms.untracked.PSet(branch = cms.untracked.string("patTriggerObjectStandAlones_slimmedPatTrigger__*"),splitLevel=cms.untracked.int32(99)),
cms.untracked.PSet(branch = cms.untracked.string("patPackedGenParticles_packedGenParticles__*"),splitLevel=cms.untracked.int32(99)),
cms.untracked.PSet(branch = cms.untracked.string("patJets_slimmedJets__*"),splitLevel=cms.untracked.int32(99)),
cms.untracked.PSet(branch = cms.untracked.string("recoVertexs_offlineSlimmedPrimaryVertices__*"),splitLevel=cms.untracked.int32(99)),
cms.untracked.PSet(branch = cms.untracked.string("recoVertexs_offlineSlimmedPrimaryVerticesWithBS__*"),splitLevel=cms.untracked.int32(99)),
cms.untracked.PSet(branch = cms.untracked.string("recoCaloClusters_reducedEgamma_reducedESClusters_*"),splitLevel=cms.untracked.int32(99)),
cms.untracked.PSet(branch = cms.untracked.string("EcalRecHitsSorted_reducedEgamma_reducedEBRecHits_*"),splitLevel=cms.untracked.int32(99)),
cms.untracked.PSet(branch = cms.untracked.string("EcalRecHitsSorted_reducedEgamma_reducedEERecHits_*"),splitLevel=cms.untracked.int32(99)),
cms.untracked.PSet(branch = cms.untracked.string("recoGenJets_slimmedGenJets__*"),splitLevel=cms.untracked.int32(99)),
cms.untracked.PSet(branch = cms.untracked.string("patJets_slimmedJetsPuppi__*"),splitLevel=cms.untracked.int32(99)),
cms.untracked.PSet(branch = cms.untracked.string("EcalRecHitsSorted_reducedEgamma_reducedESRecHits_*"),splitLevel=cms.untracked.int32(99)),
])

_phase2_hgc_extraCommands = ["keep *_slimmedElectronsHGC_*_*", "keep *_slimmedPhotonsHGC_*_*"]
from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toModify(MicroEventContentMC, outputCommands = MicroEventContentMC.outputCommands + _phase2_hgc_extraCommands)

_phase2_timing_extraCommands = ["keep *_offlineSlimmedPrimaryVertices4D_*_*"]
from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_timing.toModify(MicroEventContentMC, outputCommands = MicroEventContentMC.outputCommands + _phase2_timing_extraCommands)
