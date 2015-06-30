import FWCore.ParameterSet.Config as cms

# set cabling by hand (typically for testbeam)
Phase2TrackerCabling = cms.ESSource("Phase2TrackerCablingCfgESSource",
    modules = cms.VPSet(
                 cms.PSet( # Phase2 tracker module connection
                   moduleType=cms.string("2S"),
                   detid=cms.uint32(50000), 
                   gbtid=cms.uint32(10), 
                   fedid=cms.uint32(51), 
                   fedch=cms.uint32(0), 
                   powerGroup=cms.uint32(0), 
                   coolingLoop=cms.uint32(0)
                 ),
                 cms.PSet( # Phase2 tracker module connection
                   moduleType=cms.string("2S"),
                   detid=cms.uint32(51000), 
                   gbtid=cms.uint32(11), 
                   fedid=cms.uint32(51), 
                   fedch=cms.uint32(1), 
                   powerGroup=cms.uint32(0), 
                   coolingLoop=cms.uint32(0)
                 ),
              )
)
