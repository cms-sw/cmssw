import FWCore.ParameterSet.Config as cms

pythia8aMCatNLOSettingsBlock = cms.PSet(
    pythia8aMCatNLOSettings = cms.vstring(
      'SpaceShower:pTmaxMatch = 1',
      'SpaceShower:pTmaxFudge = 1',
      'SpaceShower:MEcorrections = off',
      'TimeShower:pTmaxMatch = 1',
      'TimeShower:pTmaxFudge = 1',
      'TimeShower:MEcorrections = off',
      'TimeShower:globalRecoil = on',
      'TimeShower:limitPTmaxGlobal = on',
      'TimeShower:nMaxGlobalRecoil = 1',
      'TimeShower:globalRecoilMode = 2',
      'TimeShower:nMaxGlobalBranch = 1',
    )
)
