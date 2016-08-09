# cmssw python configuration file to test HCAL RecHit dumper

import FWCore.ParameterSet.Config as cms


#ifname="file:/afs/cern.ch/user/h/halil/public/HCALUpgradeSW/test_hcalrhdumper.root"
ifname="file:/afs/cern.ch/user/h/halil/public/HCALUpgradeSW/step3_old.root"
#ifname="file:/afs/cern.ch/user/h/halil/public/HCALUpgradeSW/step3_new.root"

process = cms.Process("HCALDump")

process.maxEvents = cms.untracked.PSet(
            input = cms.untracked.int32 (2)

)

process.source = cms.Source("PoolSource",
 fileNames = cms.untracked.vstring(
   ifname
 )
)

# Add the indices of the bits you want printed out to the "bits" list below.
# Bits are printed in the same order as they appear on the list.
# Bit indices range from 0 to 127. What's printed is:
# flags - aux - auxHBHE - auxPhase1 (32 bits each)
# ^127^96   ^64       ^32         ^0
# A -1 for bit index prints a separator
# When dumping non-HBHE rechits, HBHE specific bits are printed as 0

process.hitDumper = cms.EDAnalyzer("HcalRecHitDump",
                                   # hbhePrefix=cms.untracked.string("!hbhe!"),
                                   # hfPrefix=cms.untracked.string("!hf!"),
                                   bits = cms.untracked.vint32(127,96,-1,95,64,-1,63,32,-1,31,0 )
                                   )

process.p = cms.Path(process.hitDumper)
