import FWCore.ParameterSet.Config as cms

process = cms.Process('VALID')

process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.PyReleaseValidation.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']

process.source = cms.Source('EmptySource')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.valid = cms.EDAnalyzer("ValidateGeometry",
                               #infileName = cms.untracked.string('/afs/cern.ch/cms/fireworks/release/cmsShow36/cmsGeom10.root'),
                               infileName = cms.untracked.string('/afs/cern.ch/user/y/yana/public/cmsRecoGeom1.root'),
                               outfileName = cms.untracked.string('validateGeometry.root')
                               )

process.p = cms.Path(process.valid)
