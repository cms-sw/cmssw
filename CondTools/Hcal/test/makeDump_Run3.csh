#!/bin/csh
cmsenv

if ("$4" == "test") then
     echo "connectstring = sqlite_file:test.db"
     set connectstring = sqlite_file:test.db
else  if ( $#argv == 3 ) then
    echo "connectstring = frontier://FrontierProd/CMS_CONDITIONS"
    set connectstring = frontier://FrontierProd/CMS_CONDITIONS
else
    echo "connectstring = $4"
    set connectstring = $4
endif

cat >! temp_dump_cfg.py <<%

# Usage example:   
# ./makeDump_Run3.csh ElectronicsMap HcalElectronicsMap_v9.0_hlt 309055 frontier://FrontierProd/CMS_CONDITIONS
import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras
process = cms.Process("DUMP",eras.Run3)

process.load("Configuration.Geometry.GeometryExtended2021_cff")
process.load("Configuration.Geometry.GeometryExtended2021Reco_cff")
  
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32($3)
)

process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = '$connectstring'

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
        process.CondDB,
        toGet = cms.VPSet(cms.PSet(record = cms.string("Hcal$1Rcd"),
                                   tag = cms.string("$2")
                                  )                        
                         )
)

process.dumpcond = cms.EDAnalyzer("HcalDumpConditions",
                                  dump = cms.untracked.vstring("$1")
                                  )

process.p = cms.Path(process.dumpcond)

%
cmsRun temp_dump_cfg.py
rm temp_dump_cfg.py

