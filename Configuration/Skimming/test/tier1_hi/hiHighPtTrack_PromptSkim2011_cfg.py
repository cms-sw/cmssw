import FWCore.ParameterSet.Config as cms
process = cms.Process("HIGHPTSKIM")

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContentHeavyIons_cff')

process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
        'file:/mnt/hadoop/cms/store/hidata/HIRun2010/HIAllPhysics/RECO/SDmaker_3SD_1CS_PDHIAllPhysicsZSv2_SD_JetHI-v1/0000/A8934EC1-904B-E011-862C-003048F17528.root'
)
)

# =============== Other Statements =====================
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

#Trigger Selection
### Comment out for the timing being assuming running on secondary dataset with trigger bit selected already
import HLTrigger.HLTfilters.hltHighLevel_cfi
process.hltHIHighPtTrack = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
process.hltHIHighPtTrack.HLTPaths = ['HLT_HIFullTrack20_*','HLT_HIFullTrack25_*'] # for allphysics
process.hltHIHighPtTrack.andOr = cms.bool(True)
process.hltHIHighPtTrack.throw = cms.bool(False)

process.eventFilter_step = cms.Path( process.hltHIHighPtTrack )

process.output = cms.OutputModule("PoolOutputModule",
    outputCommands = process.RECOEventContent.outputCommands,
    fileName = cms.untracked.string('hiHighPtTrack.root'),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('eventFilter_step')),
    dataset = cms.untracked.PSet(
      dataTier = cms.untracked.string('RECO'),
      filterName = cms.untracked.string('hiHighPtTrack'))
)

process.output_step = cms.EndPath(process.output)

process.schedule = cms.Schedule(
    process.eventFilter_step,
    process.output_step
)
