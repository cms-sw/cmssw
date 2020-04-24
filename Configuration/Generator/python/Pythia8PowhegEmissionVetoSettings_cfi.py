import FWCore.ParameterSet.Config as cms

pythia8PowhegEmissionVetoSettingsBlock = cms.PSet(
    pythia8PowhegEmissionVetoSettings = cms.vstring(
          'POWHEG:veto = 1',
          'POWHEG:pTdef = 1',
          'POWHEG:emitted = 0',
          'POWHEG:pTemt = 0',
          'POWHEG:pThard = 0',
          'POWHEG:vetoCount = 100',
          'SpaceShower:pTmaxMatch = 2',
          'TimeShower:pTmaxMatch = 2',
    )
)
