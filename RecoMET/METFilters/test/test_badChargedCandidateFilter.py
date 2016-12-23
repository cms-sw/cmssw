import FWCore.ParameterSet.Config as cms
process = cms.Process("test")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.source = cms.Source(
    'PoolSource',
    #RECO
    #fileNames = cms.untracked.vstring('root://eoscms.cern.ch//store/group/phys_jetmet/MetScanners/bobak_pickevents.root')
    #fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/m/mzeinali/public/skim_9.root')
    #fileNames = cms.untracked.vstring( 'file:lowMet_failBadChargedHadronFilter.root')
    #miniAOD
    #fileNames = cms.untracked.vstring('root://eoscms.cern.ch//store/group/phys_jetmet/MetScanners/bobak_pickevents_miniAOD.root ')
    fileNames = cms.untracked.vstring('root://eoscms.cern.ch//store/mc/RunIISpring16MiniAODv1/SMS-T1tttt_mGluino-1500_mLSP-100_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/PUSpring16_80X_mcRun2_asymptotic_2016_v3-v1/60000/26C76166-0FFE-E511-BA96-0025905D1D60.root')
    #fileNames = cms.untracked.vstring( "file:/afs/cern.ch/user/s/schoef/eos/cms/store/mc/RunIISpring16MiniAODv1/SMS-T1tttt_mGluino-1500_mLSP-100_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/PUSpring16_80X_mcRun2_asymptotic_2016_v3-v1/60000/26C76166-0FFE-E511-BA96-0025905D1D60.root")
    #fileNames = cms.untracked.vstring( 'file:/data/rschoefbeck/local/SMS-T1tttt_mGluino-1500_mLSP-100_TuneCUETP8M1_13_MINIAODSIM.root')
    )

process.load('Configuration.StandardSequences.Services_cff')
process.load('RecoMET.METFilters.BadChargedCandidateFilter_cfi')
process.load('RecoMET.METFilters.BadChargedCandidateSummer16Filter_cfi')

## for miniAOD running
process.BadChargedCandidateFilter.muons = cms.InputTag("slimmedMuons")
process.BadChargedCandidateFilter.PFCandidates = cms.InputTag("packedPFCandidates")

process.BadChargedCandidateFilter.debug = cms.bool(True)

## for miniAOD running (ICHEP)
process.BadChargedCandidateSummer16Filter.muons = cms.InputTag("slimmedMuons")
process.BadChargedCandidateSummer16Filter.PFCandidates = cms.InputTag("packedPFCandidates")

process.BadChargedCandidateSummer16Filter.debug = cms.bool(True)

process.out = cms.OutputModule("PoolOutputModule",
     fileName = cms.untracked.string('histo.root'),
     outputCommands = cms.untracked.vstring('keep *') 
)
#
# RUN!
#
process.run = cms.Path(
  process.BadChargedCandidateFilter * process.BadChargedCandidateSummer16Filter 
)

process.outpath = cms.EndPath(process.out)
