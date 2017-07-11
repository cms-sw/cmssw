import FWCore.ParameterSet.Config as cms

MicroEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_slimmedPhotons_*_*',
        'keep *_slimmedElectrons_*_*',
        'keep *_slimmedMuons_*_*',
        'keep *_slimmedTaus_*_*',
        'keep *_slimmedTausBoosted_*_*',
        'keep *_slimmedJets_*_*',
        'keep *_slimmedJetsAK8_*_*',
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

        'drop *_*_caloTowers_*',
        'drop *_*_pfCandidates_*',
        'drop *_*_genJets_*',

        'keep *_offlineBeamSpot_*_*',
        'keep *_offlineSlimmedPrimaryVertices_*_*',
        'keep patPackedCandidates_packedPFCandidates_*_*',
        'keep *_isolatedTracks_*_*',
        # low energy conversions for BPH
        'keep *_oniaPhotonCandidates_*_*',

        'keep *_bunchSpacingProducer_*_*',

        'keep double_fixedGridRhoAll__*',
        'keep double_fixedGridRhoFastjetAll__*',
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
        'keep CTPPSLocalTrackLites_ctppsLocalTrackLiteProducer_*_*'
    )
)
MicroEventContentMC = cms.PSet(
    outputCommands = cms.untracked.vstring(MicroEventContent.outputCommands)
)
MicroEventContentMC.outputCommands += [
        #'keep *_slimmedGenJets*_*_*',
        'keep *_slimmedGenJets_*_*',
        'keep *_slimmedGenJetsAK8_*_*',
        'keep patPackedGenParticles_packedGenParticles_*_*',
        'keep recoGenParticles_prunedGenParticles_*_*',
        'keep LHEEventProduct_*_*_*',
        'keep PileupSummaryInfos_slimmedAddPileupInfo_*_*',
        'keep GenFilterInfo_*_*_*',
        'keep GenLumiInfoHeader_generator_*_*',
        'keep GenLumiInfoProduct_*_*_*',
        'keep GenEventInfoProduct_generator_*_*',
        # RUN
        'keep LHERunInfoProduct_*_*_*',
        'keep GenRunInfoProduct_*_*_*',
        'keep L1GtTriggerMenuLite_l1GtTriggerMenuLite__*',
]

MiniAODOverrideBranchesSplitLevel = cms.untracked.VPSet( [
cms.untracked.PSet(branch = cms.untracked.string("patMuons_slimmedMuons__*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("patElectrons_slimmedElectrons__*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("patTaus_slimmedTaus__*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("patPhotons_slimmedPhotons__*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("patTaus_slimmedTausBoosted__*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("patCompositeCandidates_oniaPhotonCandidates_conversions_*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("recoSuperClusters_reducedEgamma_reducedSuperClusters_*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("recoConversions_reducedEgamma_reducedConversions_*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("patPackedCandidates_lostTracks__*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("patMETs_slimmedMETs__*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("patMETs_slimmedMETsPuppi__*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("patMETs_slimmedMETsNoHF__*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("recoVertexCompositePtrCandidates_slimmedKshortVertices__*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("patJets_slimmedJetsAK8__*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("HcalNoiseSummary_hcalnoise__*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("patJets_slimmedJetsAK8PFPuppiSoftDropPacked_SubJets_*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("patIsolatedTracks_isolatedTracks__*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("GenEventInfoProduct_generator__*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("l1tEGammaBXVector_caloStage2Digis_EGamma_*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("l1tEtSumBXVector_caloStage2Digis_EtSum_*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("recoGenJets_slimmedGenJetsAK8__*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("recoVertexCompositePtrCandidates_slimmedLambdaVertices__*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("GlobalAlgBlkBXVector_gtStage2Digis__*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("l1tMuonBXVector_gmtStage2Digis_Muon_*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("patPackedCandidates_lostTracks_eleTracks_*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("recoConversions_reducedEgamma_reducedSingleLegConversions_*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("recoGsfElectronCores_reducedEgamma_reducedGedGsfElectronCores_*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("recoPhotonCores_reducedEgamma_reducedGedPhotonCores_*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("recoCSCHaloData_CSCHaloData__*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("recoBeamHaloSummary_BeamHaloSummary__*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("GlobalExtBlkBXVector_gtStage2Digis__*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("recoBeamSpot_offlineBeamSpot__*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("l1extraL1EtMissParticles_l1extraParticles_MET_*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("l1extraL1EtMissParticles_l1extraParticles_MHT_*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("l1extraL1HFRingss_l1extraParticles__*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("l1extraL1EmParticles_l1extraParticles_NonIsolated_*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("l1extraL1JetParticles_l1extraParticles_IsoTau_*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("l1extraL1JetParticles_l1extraParticles_Forward_*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("l1extraL1JetParticles_l1extraParticles_Central_*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("l1extraL1EmParticles_l1extraParticles_Isolated_*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("l1extraL1MuonParticles_l1extraParticles__*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("l1extraL1JetParticles_l1extraParticles_Tau_*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("LumiScalerss_scalersRawToDigi__*"),splitLevel=cms.untracked.int32(0)),
cms.untracked.PSet(branch = cms.untracked.string("patPhotons_slimmedOOTPhotons__*"),splitLevel=cms.untracked.int32(0)),
])
