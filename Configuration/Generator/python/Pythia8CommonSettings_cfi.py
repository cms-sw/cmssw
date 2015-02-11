import FWCore.ParameterSet.Config as cms

pythia8CommonSettingsBlock = cms.PSet(
    pythia8CommonSettings = cms.vstring(
      'Tune:preferLHAPDF = 2',
      'Main:timesAllowErrors = 10000',
      'Check:epTolErr = 0.01',
      'Beams:setProductionScalesFromLHEF = off',
      'SLHA:keepSM = on',
      'SLHA:minMassSM = 1000.',
      'ParticleDecays:limitTau0 = on',
      'ParticleDecays:tau0Max = 10',
      'ParticleDecays:allowPhotonRadiation = on',
    )
)
