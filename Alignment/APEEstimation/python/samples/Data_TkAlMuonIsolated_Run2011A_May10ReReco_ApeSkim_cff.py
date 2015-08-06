import FWCore.ParameterSet.Config as cms

maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)


readFiles.extend( [

       '/store/caf/user/hauk/data/DoubleMu/Run2011A_May10ReReco/apeSkim1.root',
       '/store/caf/user/hauk/data/DoubleMu/Run2011A_May10ReReco/apeSkim2.root',
       '/store/caf/user/hauk/data/DoubleMu/Run2011A_May10ReReco/apeSkim3.root',

] )
