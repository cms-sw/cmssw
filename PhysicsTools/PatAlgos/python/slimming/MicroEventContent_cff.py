import FWCore.ParameterSet.Config as cms

MicroEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_slimmedPhotons_*_*',
        'keep *_slimmedElectrons_*_*',
        'keep *_slimmedMuons_*_*',
        'keep *_slimmedTaus_*_*',
        'keep *_slimmedJets_*_*',
        'keep *_slimmedJetsAK8_*_*',
        'keep *_slimmedJetsPuppi_*_*',
        #'keep *_slimmedMETs*_*_*',
        'keep *_slimmedMETs_*_*',
        'keep *_slimmedMETsPuppi_*_*',
        'keep *_slimmedSecondaryVertices_*_*',
        #'keep *_cmsTopTaggerMap_*_*',
        #'keep *_slimmedJetsAK8PFCHSSoftDropSubjets_*_*',
        #'keep *_slimmedJetsCMSTopTagCHSSubjets_*_*',
        'keep *_slimmedJetsAK8PFCHSSoftDropPacked_SubJets_*',
        'keep *_slimmedJetsCMSTopTagCHSPacked_SubJets_*',
        #'keep *_packedPatJetsAK8_*_*',
        ## add extra METs

        #'keep recoPhotonCores_reducedEgamma_*_*',
        'keep recoPhotonCores_reducedEgamma_reducedGedPhotonCores_*',
        'keep recoGsfElectronCores_reducedEgamma_reducedGedGsfElectronCores_*',
        'keep recoConversions_reducedEgamma_reducedConversions_*',
        'keep recoConversions_reducedEgamma_reducedSingleLegConversions_*',
        'keep recoSuperClusters_reducedEgamma_reducedSuperClusters_*',
        'keep recoCaloClusters_reducedEgamma_reducedEBEEClusters_*',
        'keep recoCaloClusters_reducedEgamma_reducedESClusters_*',
        'keep EcalRecHitsSorted_reducedEgamma_reducedEBRecHits_*',
        'keep EcalRecHitsSorted_reducedEgamma_reducedEERecHits_*',
        'keep EcalRecHitsSorted_reducedEgamma_reducedESRecHits_*',


        'drop *_*_caloTowers_*',
        'drop *_*_pfCandidates_*',
        'drop *_*_genJets_*',

        'keep *_offlineBeamSpot_*_*',
        'keep *_offlineSlimmedPrimaryVertices_*_*',
        'keep patPackedCandidates_packedPFCandidates_*_*',

        #'keep double_fixedGridRho*__*',
        'keep double_fixedGridRhoAll__*',
        'keep double_fixedGridRhoFastjetAll__*',
        'keep double_fixedGridRhoFastjetAllCalo__*',
        'keep double_fixedGridRhoFastjetCentralCalo__*',
        'keep double_fixedGridRhoFastjetCentralChargedPileUp__*',
        'keep double_fixedGridRhoFastjetCentralNeutral__*',

        'keep *_selectedPatTrigger_*_*',
        'keep patPackedTriggerPrescales_patTrigger__*',
        'keep *_l1extraParticles_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_TriggerResults_*_HLT',
        'keep *_TriggerResults_*_*', # for MET filters (a catch all for the moment, but ideally it should be only the current process)
        'keep patPackedCandidates_lostTracks_*_*',
        'keep HcalNoiseSummary_hcalnoise__*',
        'keep *_caTopTagInfosPAT_*_*'
    )
)
MicroEventContentMC = cms.PSet(
    outputCommands = cms.untracked.vstring(MicroEventContent.outputCommands)
)
MicroEventContentMC.outputCommands += [
        'keep *_slimmedGenJets*_*_*',
        'keep patPackedGenParticles_packedGenParticles_*_*',
        'keep recoGenParticles_prunedGenParticles_*_*',
        'keep LHEEventProduct_*_*_*',
        'keep PileupSummaryInfos_slimmedAddPileupInfo_*_*',
        'keep GenFilterInfo_*_*_*',
        'keep GenLumiInfoProduct_*_*_*',
        'keep GenEventInfoProduct_generator_*_*',
        # RUN
        'keep LHERunInfoProduct_*_*_*',
        'keep GenRunInfoProduct_*_*_*',
        'keep L1GtTriggerMenuLite_l1GtTriggerMenuLite__*',
]
