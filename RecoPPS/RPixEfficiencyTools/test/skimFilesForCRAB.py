import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('reRECO',eras.Run2_2018)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('FWCore.ParameterSet.Types')


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        ),
    duplicateCheckMode = cms.untracked.string("checkAllFilesOpened"),
    # skipBadFiles = cms.untracked.bool(True),
)

from HLTrigger.HLTfilters.hltHighLevel_cfi import *
process.triggerSelection  = hltHighLevel.clone(TriggerResultsTag = "TriggerResults::HLT", HLTPaths = ['HLT_Ele32_WPTight_Gsf_L1DoubleEG_v*','HLT_Ele35_WPTight_Gsf_v*'])

from RecoPPS.Configuration.RecoPPS_EventContent_cff import RecoPPSAOD
process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('ReReco_2018B1.root'),
    outputCommands = cms.untracked.vstring("drop *",
    	"keep *_ctppsProtons_*_RECO",
        "keep *_ctppsLocalTrackLiteProducer_*_RECO",
        "keep *_ctppsPixelRecHits_*_RECO",
        "keep *_ctppsPixelLocalTracks_*_RECO",
        "keep *_ctppsPixelClusters_*_RECO",
    	)
)
# process.output.outputCommands.extend(RecoPPSAOD.outputCommands)

# Path and EndPath definitions
process.output_step = cms.EndPath(process.output)

# Schedule definition
process.schedule = cms.Schedule(process.output_step)
