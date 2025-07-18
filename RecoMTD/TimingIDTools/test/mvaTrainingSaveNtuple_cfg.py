import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9

process = cms.Process("SaveNtuple",Phase2C17I13M9)

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.Geometry.GeometryExtendedRun4D110Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T33', '')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.TFileService = cms.Service("TFileService",
    fileName = cms.string("output_ntuple.root"),
    closeFileFast = cms.untracked.bool(True)
)

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
                               'file:step3.root'
                            )
)

from RecoMTD.TimingIDTools.mvaTrainingNtuple_cff import mvaTrainingNtuple

process.mvaTrainingNtuple = mvaTrainingNtuple

process.p = cms.Path(process.mvaTrainingNtuple)

