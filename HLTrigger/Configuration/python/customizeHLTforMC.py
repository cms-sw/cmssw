# CSCHaloDataProducer
# not used at HLT - process.CSCHaloData.ExpectedBX = cms.int32(6)

# EcalUncalibRecHitProducer
# not used at HLT - process.ecalGlobalUncalibRecHit.doEBtimeCorrection = cms.bool(False)
# not used at HLT - process.ecalGlobalUncalibRecHit.doEEtimeCorrection = cms.bool(False)

# HcalRecAlgoESProducer
if hcalRecAlgos in process.dict():
  import RecoLocalCalo.HcalRecAlgos.RemoveAddSevLevel as HcalRemoveAddSevLevel
  HcalRemoveAddSevLevel.AddFlag(process.hcalRecAlgos, "HFDigiTime",     8)
  HcalRemoveAddSevLevel.AddFlag(process.hcalRecAlgos, "HBHEFlatNoise",  8)
  HcalRemoveAddSevLevel.AddFlag(process.hcalRecAlgos, "HBHESpikeNoise", 8)

# PFRecHitProducerHCAL
if hltParticleFlowRecHitHCAL in process.dict():
  process.hltParticleFlowRecHitHCAL.ApplyPulseDPG      = cms.bool(False)
  process.hltParticleFlowRecHitHCAL.LongShortFibre_Cut = cms.double(1000000000.0)

