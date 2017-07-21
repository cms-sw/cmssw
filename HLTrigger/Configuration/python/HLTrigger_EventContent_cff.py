import FWCore.ParameterSet.Config as cms

# EventContent for HLT related products.

# This file exports the following five EventContent blocks:
#   HLTriggerRAW  HLTriggerRECO  HLTriggerAOD (without DEBUG products)
#   HLTDebugRAW   HLTDebugFEVT                (with    DEBUG products)
#
# as these are used in Configuration/EventContent
#
HLTriggerRAW  = cms.PSet(
    outputCommands = cms.vstring( *(
        'drop *_hlt*_*_*',
        'keep FEDRawDataCollection_rawDataCollector_*_*',
        'keep FEDRawDataCollection_source_*_*',
        'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
        'keep edmTriggerResults_*_*_*',
        'keep triggerTriggerEvent_*_*_*'
    ) )
)

HLTriggerRECO = cms.PSet(
    outputCommands = cms.vstring( *(
        'drop *_hlt*_*_*',
        'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
        'keep edmTriggerResults_*_*_*',
        'keep triggerTriggerEvent_*_*_*'
    ) )
)

HLTriggerAOD  = cms.PSet(
    outputCommands = cms.vstring( *(
        'drop *_hlt*_*_*',
        'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
        'keep edmTriggerResults_*_*_*',
        'keep triggerTriggerEvent_*_*_*'
    ) )
)

HLTDebugRAW   = cms.PSet(
    outputCommands = cms.vstring( *(
        'drop *_hlt*_*_*',
        'keep *_hltAlCaEtaEBRechitsToDigisLowPU_*_*',
        'keep *_hltAlCaEtaEBRechitsToDigis_*_*',
        'keep *_hltAlCaEtaEERechitsToDigisLowPU_*_*',
        'keep *_hltAlCaEtaEERechitsToDigis_*_*',
        'keep *_hltAlCaEtaRecHitsFilterEEonlyRegionalLowPU_etaEcalRecHitsES_*',
        'keep *_hltAlCaEtaRecHitsFilterEEonlyRegional_etaEcalRecHitsES_*',
        'keep *_hltAlCaPi0EBRechitsToDigisLowPU_*_*',
        'keep *_hltAlCaPi0EBRechitsToDigis_*_*',
        'keep *_hltAlCaPi0EERechitsToDigisLowPU_*_*',
        'keep *_hltAlCaPi0EERechitsToDigis_*_*',
        'keep *_hltAlCaPi0RecHitsFilterEEonlyRegionalLowPU_pi0EcalRecHitsES_*',
        'keep *_hltAlCaPi0RecHitsFilterEEonlyRegional_pi0EcalRecHitsES_*',
        'keep *_hltCombinedSecondaryVertexBJetTagsCalo_*_*',
        'keep *_hltCombinedSecondaryVertexBJetTagsPF_*_*',
        'keep *_hltCscSegments_*_*',
        'keep *_hltDt4DSegments_*_*',
        'keep *_hltEcalPhiSymFilter_*_*',
        'keep *_hltEcalRecHit_*_*',
        'keep *_hltEgammaCandidates_*_*',
        'keep *_hltEgammaGsfElectrons_*_*',
        'keep *_hltEgammaGsfTracks_*_*',
        'keep *_hltElectronsVertex_*_*',
        'keep *_hltFEDSelectorLumiPixels_*_*',
        'keep *_hltFastPrimaryVertex_*_*',
        'keep *_hltGtStage2Digis_*_*',
        'keep *_hltHbhereco_*_*',
        'keep *_hltHfreco_*_*',
        'keep *_hltHoreco_*_*',
        'keep *_hltIter0ElectronsTrackSelectionHighPurity_*_*',
        'keep *_hltIter0HighPtTkMuPixelTracks_*_*',
        'keep *_hltIter0HighPtTkMuTrackSelectionHighPurity_*_*',
        'keep *_hltIter2HighPtTkMuMerged_*_*',
        'keep *_hltIter2HighPtTkMuTrackSelectionHighPurity_*_*',
        'keep *_hltIter2MergedForElectrons_*_*',
        'keep *_hltIter2Merged_*_*',
        'keep *_hltL3NoFiltersNoVtxMuonCandidates_*_*',
        'keep *_hltMuonCSCDigis_MuonCSCStripDigi_*',
        'keep *_hltMuonCSCDigis_MuonCSCWireDigi_*',
        'keep *_hltMuonDTDigis_*_*',
        'keep *_hltMuonRPCDigis_*_*',
        'keep *_hltOnlineBeamSpot_*_*',
        'keep *_hltPFJetForBtag_*_*',
        'keep *_hltPFMuonMerging_*_*',
        'keep *_hltPixelTracksElectrons_*_*',
        'keep *_hltPixelTracks_*_*',
        'keep *_hltPixelVertices_*_*',
        'keep *_hltRpcRecHits_*_*',
        'keep *_hltSelector8CentralJetsL1FastJet_*_*',
        'keep *_hltSiPixelClusters_*_*',
        'keep *_hltSiStripRawToClustersFacility_*_*',
        'keep *_hltVerticesL3_*_*',
        'keep *_hltVerticesPFFilter_*_*',
        'keep *_hltVerticesPFSelector_*_*',
        'keep FEDRawDataCollection_rawDataCollector_*_*',
        'keep FEDRawDataCollection_source_*_*',
        'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
        'keep edmTriggerResults_*_*_*',
        'keep triggerTriggerEvent_*_*_*'
    ) )
)

HLTDebugFEVT  = cms.PSet(
    outputCommands = cms.vstring( *(
        'drop *_hlt*_*_*',
        'keep *_hltAlCaEtaEBRechitsToDigisLowPU_*_*',
        'keep *_hltAlCaEtaEBRechitsToDigis_*_*',
        'keep *_hltAlCaEtaEERechitsToDigisLowPU_*_*',
        'keep *_hltAlCaEtaEERechitsToDigis_*_*',
        'keep *_hltAlCaEtaRecHitsFilterEEonlyRegionalLowPU_etaEcalRecHitsES_*',
        'keep *_hltAlCaEtaRecHitsFilterEEonlyRegional_etaEcalRecHitsES_*',
        'keep *_hltAlCaPi0EBRechitsToDigisLowPU_*_*',
        'keep *_hltAlCaPi0EBRechitsToDigis_*_*',
        'keep *_hltAlCaPi0EERechitsToDigisLowPU_*_*',
        'keep *_hltAlCaPi0EERechitsToDigis_*_*',
        'keep *_hltAlCaPi0RecHitsFilterEEonlyRegionalLowPU_pi0EcalRecHitsES_*',
        'keep *_hltAlCaPi0RecHitsFilterEEonlyRegional_pi0EcalRecHitsES_*',
        'keep *_hltCombinedSecondaryVertexBJetTagsCalo_*_*',
        'keep *_hltCombinedSecondaryVertexBJetTagsPF_*_*',
        'keep *_hltCscSegments_*_*',
        'keep *_hltDt4DSegments_*_*',
        'keep *_hltEcalPhiSymFilter_*_*',
        'keep *_hltEcalRecHit_*_*',
        'keep *_hltEgammaCandidates_*_*',
        'keep *_hltEgammaGsfElectrons_*_*',
        'keep *_hltEgammaGsfTracks_*_*',
        'keep *_hltElectronsVertex_*_*',
        'keep *_hltFEDSelectorLumiPixels_*_*',
        'keep *_hltFastPrimaryVertex_*_*',
        'keep *_hltGtStage2Digis_*_*',
        'keep *_hltHbhereco_*_*',
        'keep *_hltHfreco_*_*',
        'keep *_hltHoreco_*_*',
        'keep *_hltIter0ElectronsTrackSelectionHighPurity_*_*',
        'keep *_hltIter0HighPtTkMuPixelTracks_*_*',
        'keep *_hltIter0HighPtTkMuTrackSelectionHighPurity_*_*',
        'keep *_hltIter2HighPtTkMuMerged_*_*',
        'keep *_hltIter2HighPtTkMuTrackSelectionHighPurity_*_*',
        'keep *_hltIter2MergedForElectrons_*_*',
        'keep *_hltIter2Merged_*_*',
        'keep *_hltL3NoFiltersNoVtxMuonCandidates_*_*',
        'keep *_hltMuonCSCDigis_MuonCSCStripDigi_*',
        'keep *_hltMuonCSCDigis_MuonCSCWireDigi_*',
        'keep *_hltMuonDTDigis_*_*',
        'keep *_hltMuonRPCDigis_*_*',
        'keep *_hltOnlineBeamSpot_*_*',
        'keep *_hltPFJetForBtag_*_*',
        'keep *_hltPFMuonMerging_*_*',
        'keep *_hltPixelTracksElectrons_*_*',
        'keep *_hltPixelTracks_*_*',
        'keep *_hltPixelVertices_*_*',
        'keep *_hltRpcRecHits_*_*',
        'keep *_hltSelector8CentralJetsL1FastJet_*_*',
        'keep *_hltSiPixelClusters_*_*',
        'keep *_hltSiStripRawToClustersFacility_*_*',
        'keep *_hltVerticesL3_*_*',
        'keep *_hltVerticesPFFilter_*_*',
        'keep *_hltVerticesPFSelector_*_*',
        'keep FEDRawDataCollection_rawDataCollector_*_*',
        'keep FEDRawDataCollection_source_*_*',
        'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
        'keep edmTriggerResults_*_*_*',
        'keep triggerTriggerEvent_*_*_*'
    ) )
)

