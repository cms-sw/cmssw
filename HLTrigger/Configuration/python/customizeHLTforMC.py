import FWCore.ParameterSet.Config as cms

def customizeHLTforMC(process):
  """adapt the HLT to run on MC, instead of data
  see Configuration/StandardSequences/Reconstruction_Data_cff.py
  which does the opposite, for RECO"""

  # PFRecHitProducerHCAL
  if 'hltParticleFlowRecHitHCAL' in process.__dict__:
    process.hltParticleFlowRecHitHCAL.ApplyPulseDPG      = cms.bool(False)
    process.hltParticleFlowRecHitHCAL.LongShortFibre_Cut = cms.double(1000000000.0)

  # customise hltHbhereco to use the Method 3 time slew parametrization and response correction for Monte Carlo (PR #11091)
  if 'hltHbhereco' in process.__dict__:
    if process.hltHbhereco._TypedParameterizable__type == 'HcalHitReconstructor':
      # 2015-2016 Run 2
      process.hltHbhereco.pedestalSubtractionType = cms.int32( 1 )
      process.hltHbhereco.pedestalUpperLimit      = cms.double( 2.7 )
      process.hltHbhereco.timeSlewParsType        = cms.int32( 3 )
      # new time slew parametrisation
      process.hltHbhereco.timeSlewPars            = cms.vdouble( 12.2999, -2.19142, 0, 12.2999, -2.19142, 0, 12.2999, -2.19142, 0 )
      # old response correction, matching the 2015D 25ns data
      process.hltHbhereco.respCorrM3              = cms.double( 1.0 )
    elif process.hltHbhereco._TypedParameterizable__type == 'HBHEPhase1Reconstructor':
      # 2017 "plan 0"
      process.hltHbhereco.algorithm.respCorrM3    = cms.double( 1.0 )

  if 'hltHbhePhase1Reco' in process.__dict__:
    if process.hltHbhePhase1Reco._TypedParameterizable__type == 'HBHEPhase1Reconstructor':
      # 2017 "plan 1"
      # assume retuning the pulse shapes will give the same response
      # in data and MC for Method 2 and Method 3
      pass

  return process
