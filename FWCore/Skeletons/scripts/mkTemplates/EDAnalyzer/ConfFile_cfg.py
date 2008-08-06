import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
        'file:myfile.root'
    )
)

process.demo = cms.EDAnalyzer('anlzrname'
@example_track     , tracks = cms.untracked.InputTag('ctfWithMaterialTracks')
)

@example_histo process.TFileService = cms.Service("TFileService",
@example_histo     fileName = cms.string('histo.root')
@example_histo )

process.p = cms.Path(process.demo)
