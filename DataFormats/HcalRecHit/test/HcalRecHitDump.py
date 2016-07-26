# cmssw python configuration file to test HCAL RecHit dumper

import FWCore.ParameterSet.Config as cms

process = cms.Process("HCALDump")

process.maxEvents = cms.untracked.PSet(
            input = cms.untracked.int32 (1)

)

process.source = cms.Source("PoolSource",
 fileNames = cms.untracked.vstring(
     "file:/afs/cern.ch/user/h/halil/public/HCALUpgradeSW/test_hcalrhdumper.root"
 )
)

process.hitDumper = cms.EDAnalyzer("HcalRecHitDump")

process.p = cms.Path(process.hitDumper)
