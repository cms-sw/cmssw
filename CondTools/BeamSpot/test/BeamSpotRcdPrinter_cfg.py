import FWCore.ParameterSet.Config as cms
import os

process = cms.Process("summary")

process.MessageLogger = cms.Service( "MessageLogger",
                                     debugModules = cms.untracked.vstring( "*" ),
                                     cout = cms.untracked.PSet( threshold = cms.untracked.string( "DEBUG" ) ),
                                     destinations = cms.untracked.vstring( "cout" )
                                     )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.load("CondCore.CondDB.CondDB_cfi")
process.load("CondTools.BeamSpot.BeamSpotRcdPrinter_cfi")

### 2018 Prompt
process.BeamSpotRcdPrinter.tagName  = "BeamSpotObjects_PCL_byLumi_v0_prompt"
process.BeamSpotRcdPrinter.startIOV = 1350646955507767
process.BeamSpotRcdPrinter.endIOV   = 1406876667347162
process.BeamSpotRcdPrinter.output   = "summary2018_Prompt.txt"

### 2017 ReReco
#process.BeamSpotRcdPrinter.tagName  = "BeamSpotObjects_LumiBased_v4_offline"
#process.BeamSpotRcdPrinter.startIOV = 1275820035276801
#process.BeamSpotRcdPrinter.endIOV   = 1316235677532161

### 2018 ABC ReReco
#process.BeamSpotRcdPrinter.tagName  = "BeamSpotObjects_LumiBased_v4_offline"
#process.BeamSpotRcdPrinter.startIOV = 1354018504835073
#process.BeamSpotRcdPrinter.endIOV   = 1374668707594734

### 2018D Prompt
#process.BeamSpotRcdPrinter.tagName  = "BeamSpotObjects_PCL_byLumi_v0_prompt"
#process.BeamSpotRcdPrinter.startIOV = 1377280047710242
#process.BeamSpotRcdPrinter.endIOV   = 1406876667347162

process.p = cms.Path(process.BeamSpotRcdPrinter)
