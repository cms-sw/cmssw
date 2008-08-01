import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("RecoMET.Configuration.CaloTowersOptForMET_cff")

process.load("RecoMET.Configuration.RecoMET_cff")

process.load("RecoMET.METProducers.CaloMETSignif_cfi")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")


process.DQMStore = cms.Service("DQMStore")

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring('/store/relval/2008/6/6/RelVal-RelValZMM-1212543891-STARTUP-2nd-02/0000/9C28F593-E533-DD11-997C-000423D98BE8.root')
)

process.p = cms.Path(process.calotoweroptmaker*process.met*process.metsig)

process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string("./CaloMETSignif.root"),
                               outputCommands = cms.untracked.vstring("drop *",
                                                                      "keep *_*_*_TEST"
                                                                      )
                               )

process.e = cms.EndPath(process.out)

process.schedule = cms.Schedule(process.p)
process.schedule.append(process.e)
