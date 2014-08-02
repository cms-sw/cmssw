import FWCore.ParameterSet.Config as cms

source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/RelVal/2007/9/3/RelVal-RelValMinBias-1188839688/0002/2E7B6353-BC5A-DC11-A0B5-001617DBD5B2.root')
)

maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

