import FWCore.ParameterSet.Config as cms
process = cms.Process("test")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )
process.source = cms.Source(
    'PoolSource',
    #RECO
    fileNames = cms.untracked.vstring('root://eoscms.cern.ch//store/group/phys_jetmet/MetScanners/bobak_pickevents.root')
    #fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/k/khurana/public/JME-METScanning/MET/HighpTMuon/pickevents.root')
    #miniAOD
    #fileNames = cms.untracked.vstring('root://eoscms.cern.ch//store/group/phys_jetmet/MetScanners/bobak_pickevents_miniAOD.root ')
    )


process.load('Configuration.StandardSequences.Services_cff')
process.load('RecoMET.METFilters.BadChargedCandidateFilter_cfi')

## for miniAOD running
#process.BadChargedCandidateFilter.muons = cms.InputTag("slimmedMuons")
#process.BadChargedCandidateFilter.PFCandidates = cms.InputTag("packedPFCandidates")

process.BadChargedCandidateFilter.debug = cms.bool(True)

process.out = cms.OutputModule("PoolOutputModule",
     fileName = cms.untracked.string('histo.root'),
     outputCommands = cms.untracked.vstring('keep *') 
)
#
# RUN!
#
process.run = cms.Path(
  process.BadChargedCandidateFilter
)

process.outpath = cms.EndPath(process.out)
