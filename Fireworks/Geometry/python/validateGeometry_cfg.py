import FWCore.ParameterSet.Config as cms

process = cms.Process('VALID')

process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'START3X_V21::All'

process.source = cms.Source('EmptySource')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.valid = cms.EDAnalyzer("ValidateGeometry",
                               infileName = cms.untracked.string('/afs/cern.ch/cms/fireworks/release/cmsShow36/cmsGeom10.root'),
                               outfileName = cms.untracked.string('validateGeometry.root')
                               )

process.p = cms.Path(process.valid)
