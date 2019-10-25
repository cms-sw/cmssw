import FWCore.ParameterSet.Config as cms
process = cms.Process("ME0SegmentMatchingLocalTest")


## Standard sequence
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')

process.load('Configuration.Geometry.GeometryExtended2015MuonGEMDevReco_cff')
process.load('Configuration.Geometry.GeometryExtended2015MuonGEMDev_cff')

process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

## TrackingComponentsRecord required for matchers
process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi')
process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi')

## global tag for 2021 upgrade studies
from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2021', '')
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')

process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.me0Customs
from SLHCUpgradeSimulations.Configuration.me0Customs import customise 
process = customise(process)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:///somewhere/simevent.root') ##/somewhere/simevent.root" }
)


#process.load('RecoLocalMuon.GEMRecHit.me0RecHits_cfi')
#process.load('RecoLocalMuon.GEMSegments.me0Segments_cfi')
process.load('RecoMuon.MuonIdentification.me0MuonReco_cff')

#process.p = cms.Path(process.me0RecHits*process.me0Segments*process.me0MuonReco)
process.p = cms.Path(process.me0MuonReco)
#process.p = cms.Path(process.me0RecHits*process.me0Segments)

process.PoolSource.fileNames = [
    
    'file:out_local_reco_me0segment.root'
]


process.o1 = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_me0SegmentMatcher_*_*'
        #'drop *',
        ##'keep *_me0SegmentMatching_*_*',
        #'keep *_me0MuonConverting_*_*',
        ),
                              fileName = cms.untracked.string('out_me0Reco.root')
                              )

process.outpath = cms.EndPath(process.o1)
