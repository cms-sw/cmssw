# cmssw python configuration file to test HCAL RecHit dumper

import FWCore.ParameterSet.Config as cms


#ifname="file:/afs/cern.ch/user/h/halil/public/HCALUpgradeSW/test_hcalrhdumper.root"
ifname="file:/afs/cern.ch/user/h/halil/public/HCALUpgradeSW/step3_old.root"
#ifname="file:/afs/cern.ch/user/h/halil/public/HCALUpgradeSW/step3_new.root"

process = cms.Process("HCALDump")

process.maxEvents = cms.untracked.PSet(
            input = cms.untracked.int32 (1)

)

process.source = cms.Source("PoolSource",
 fileNames = cms.untracked.vstring(
   ifname
 )
)

process.hitDumper = cms.EDAnalyzer("HcalRecHitDump",
                                   flagsb = cms.untracked.vint32( 21,31,30,29,28,27,26,25,24,23,22,21,20,-1, 0 ), # -1 prints a separator 
                                   auxb = cms.untracked.vint32( 1, -1, 5 ),
                                   auxHBHEb = cms.untracked.vint32( 1, -1, 5 ),
                                   auxPhase1b = cms.untracked.vint32( 1, -1, 5 )
                                   )


process.p = cms.Path(process.hitDumper)
