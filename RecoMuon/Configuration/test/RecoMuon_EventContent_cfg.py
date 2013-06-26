import FWCore.ParameterSet.Config as cms

process = cms.Process("testRecoMuonEventContent")
process.load("RecoMuon.Configuration.RecoMuon_EventContent_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/2008/6/20/RelVal-RelValWM-1213921089-STARTUP_V1-2nd/0000/12727E22-B83E-DD11-952C-000423D999CA.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *'),
    fileName = cms.untracked.string('fevt.root')
)

process.RECO = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *'),
    fileName = cms.untracked.string('reco.root')
)

process.AOD = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *'),
    fileName = cms.untracked.string('aod.root')
)

process.printEventNumber = cms.OutputModule("AsciiOutputModule")

process.this_is_the_end = cms.EndPath(process.FEVT*process.RECO*process.AOD*process.printEventNumber)
process.FEVT.outputCommands.extend(process.RecoMuonFEVT.outputCommands)
process.RECO.outputCommands.extend(process.RecoMuonRECO.outputCommands)
process.AOD.outputCommands.extend(process.RecoMuonAOD.outputCommands)


