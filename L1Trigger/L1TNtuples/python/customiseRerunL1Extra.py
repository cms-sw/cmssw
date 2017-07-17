import FWCore.ParameterSet.Config as cms

def customise(process):

    process.load('EventFilter.L1TRawToDigi.caloStage1Digis_cfi')

    process.load('L1Trigger.L1TCalorimeter.simCaloStage1FinalDigis_cfi')
    process.simCaloStage1FinalDigis.InputCollection = cms.InputTag("caloStage1Digis")
    process.simCaloStage1FinalDigis.InputRlxTauCollection = cms.InputTag("caloStage1Digis:rlxTaus")
    process.simCaloStage1FinalDigis.InputIsoTauCollection = cms.InputTag("caloStage1Digis:isoTaus")
    process.simCaloStage1FinalDigis.InputPreGtJetCollection = cms.InputTag("caloStage1Digis")
    process.simCaloStage1FinalDigis.InputHFSumsCollection = cms.InputTag("caloStage1Digis:HFRingSums")
    process.simCaloStage1FinalDigis.InputHFCountsCollection = cms.InputTag("caloStage1Digis:HFBitCounts")
    
    process.load('L1Trigger.L1TCalorimeter.simCaloStage1LegacyFormatDigis_cfi')
    process.simCaloStage1LegacyFormatDigis.InputRlxTauCollection = cms.InputTag("simCaloStage1FinalDigis:rlxTaus")
    process.simCaloStage1LegacyFormatDigis.InputIsoTauCollection = cms.InputTag("simCaloStage1FinalDigis:isoTaus")
    process.simCaloStage1LegacyFormatDigis.InputHFSumsCollection = cms.InputTag("simCaloStage1FinalDigis:HFRingSums")
    process.simCaloStage1LegacyFormatDigis.InputHFCountsCollection = cms.InputTag("simCaloStage1FinalDigis:HFBitCounts")

    process.load('L1Trigger.L1ExtraFromDigis.l1extraParticles_cfi')
    
    process.l1extraParticles.isolatedEmSource    = cms.InputTag("simCaloStage1LegacyFormatDigis","isoEm")
    process.l1extraParticles.nonIsolatedEmSource = cms.InputTag("simCaloStage1LegacyFormatDigis","nonIsoEm")    
    process.l1extraParticles.forwardJetSource = cms.InputTag("simCaloStage1LegacyFormatDigis","forJets")
    process.l1extraParticles.centralJetSource = cms.InputTag("simCaloStage1LegacyFormatDigis","cenJets")
    process.l1extraParticles.tauJetSource     = cms.InputTag("simCaloStage1LegacyFormatDigis","tauJets")
    process.l1extraParticles.isoTauJetSource  = cms.InputTag("simCaloStage1LegacyFormatDigis","isoTauJets")
    process.l1extraParticles.etTotalSource = cms.InputTag("simCaloStage1LegacyFormatDigis")
    process.l1extraParticles.etHadSource   = cms.InputTag("simCaloStage1LegacyFormatDigis")
    process.l1extraParticles.etMissSource  = cms.InputTag("simCaloStage1LegacyFormatDigis")
    process.l1extraParticles.htMissSource  = cms.InputTag("simCaloStage1LegacyFormatDigis")
    process.l1extraParticles.hfRingEtSumsSource    = cms.InputTag("simCaloStage1LegacyFormatDigis")
    process.l1extraParticles.hfRingBitCountsSource = cms.InputTag("simCaloStage1LegacyFormatDigis")
    
    process.rerunL1Extra = cms.Path(
        process.caloStage1Digis +
        process.simCaloStage1FinalDigis + 
        process.simCaloStage1LegacyFormatDigis +
        process.l1extraParticles
        )
    
    process.schedule.append(process.rerunL1Extra)

    return process
