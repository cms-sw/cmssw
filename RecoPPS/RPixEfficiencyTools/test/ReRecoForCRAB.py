import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('reRECO',eras.ctpps_2016)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('FWCore.ParameterSet.Types')


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        # '/store/data/Run2017B/DoubleEG/AOD/17Nov2017-v1/30000/B6A6D63A-A0D5-E711-AACE-E0071B7A7840.root'
        ),
    duplicateCheckMode = cms.untracked.string("checkAllFilesOpened"),
    # skipBadFiles = cms.untracked.bool(True),
)

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '106X_dataRun2_v21', '')

from RecoPPS.Configuration.RecoPPS_cff import *
process.load("RecoPPS.Configuration.RecoPPS_cff")
process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")

from HLTrigger.HLTfilters.hltHighLevel_cfi import *
process.triggerSelection  = hltHighLevel.clone(TriggerResultsTag = "TriggerResults::HLT", HLTPaths = ['HLT_Ele32_WPTight_Gsf_L1DoubleEG_v*','HLT_Ele35_WPTight_Gsf_v*'])

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('ReReco_2018D.root'),
    outputCommands = cms.untracked.vstring("drop *",
        "keep *LocalTrack*_*_*_reRECO",
        "keep *Protons*_*_*_reRECO",
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring(
          'path'
        )
    )
)

# Path and EndPath definitions

process.output_step = cms.EndPath(process.output)

# processing sequence
process.path = cms.Path(
  process.triggerSelection *
  process.totemRPUVPatternFinder *
  process.totemRPLocalTrackFitter *

  process.ctppsDiamondRecHits *
  process.ctppsDiamondLocalTracks *

  process.ctppsPixelLocalTracks *

  process.ctppsLocalTrackLiteProducer *

  process.ctppsProtons
  
)

# # Schedule definition
process.schedule = cms.Schedule(
process.path, process.output_step)
