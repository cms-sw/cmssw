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
                               fileName = cms.untracked.string('/afs/cern.ch/user/m/mccauley/cmsGeom11.root'),
                               #fileName = cms.untracked.string('/afs/cern.ch/cms/fireworks/beta/cmsShow36-beta-2/cmsGeom10.root'),
                               tolerance = cms.untracked.double(0.1)
                               )


process.p = cms.Path(process.valid)
