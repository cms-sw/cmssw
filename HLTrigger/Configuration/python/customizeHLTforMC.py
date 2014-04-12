import FWCore.ParameterSet.Config as cms

def customizeHLTforMC(process):
  """adapt the HLT to run on MC, instead of data
  see Configuration/StandardSequences/Reconstruction_Data_cff.py
  which does the opposite, for RECO"""

  # CSCHaloDataProducer - not used at HLT
  #if 'CSCHaloData' in process.__dict__:
  #  process.CSCHaloData.ExpectedBX = cms.int32(6)

  # HcalRecAlgoESProducer - these flags are not used at HLT (they should stay set to the default value for both data and MC)
  #if 'hcalRecAlgos' in process.__dict__:
  #  import RecoLocalCalo.HcalRecAlgos.RemoveAddSevLevel as HcalRemoveAddSevLevel
  #  HcalRemoveAddSevLevel.AddFlag(process.hcalRecAlgos, "HFDigiTime",     8)
  #  HcalRemoveAddSevLevel.AddFlag(process.hcalRecAlgos, "HBHEFlatNoise",  8)
  #  HcalRemoveAddSevLevel.AddFlag(process.hcalRecAlgos, "HBHESpikeNoise", 8)

  # PFRecHitProducerHCAL
  if 'hltParticleFlowRecHitHCAL' in process.__dict__:
    process.hltParticleFlowRecHitHCAL.ApplyPulseDPG      = cms.bool(False)
    process.hltParticleFlowRecHitHCAL.LongShortFibre_Cut = cms.double(1000000000.0)

  return process
