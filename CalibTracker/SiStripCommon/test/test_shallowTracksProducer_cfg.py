from CalibTracker.SiStripCommon.shallowTree_test_template import *
process.TFileService.fileName = 'test_shallowTracksProducer.root'

process.load('CalibTracker.SiStripCommon.ShallowTracksProducer_cfi')
process.testTree = cms.EDAnalyzer(
   "ShallowTree",
   outputCommands = cms.untracked.vstring(
      'drop *',
      'keep *_shallowTracks_*_*',
      )
   )
process.p = cms.Path(process.shallowTracks*process.testTree)
# foo bar baz
# rgW3b2h3YJ4rW
# pTFRI3LnupGLe
