import FWCore.ParameterSet.Config as cms

maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)


readFiles.extend( [

       '/store/caf/user/hauk/data/DoubleMu/Run2011A_PromptV4/apeSkim1.root',
       '/store/caf/user/hauk/data/DoubleMu/Run2011A_PromptV4/apeSkim2.root',
       '/store/caf/user/hauk/data/DoubleMu/Run2011A_PromptV4/apeSkim3.root',
       '/store/caf/user/hauk/data/DoubleMu/Run2011A_PromptV4/apeSkim4.root',
       '/store/caf/user/hauk/data/DoubleMu/Run2011A_PromptV4/apeSkim5.root',
       '/store/caf/user/hauk/data/DoubleMu/Run2011A_PromptV4/apeSkim6.root',
       '/store/caf/user/hauk/data/DoubleMu/Run2011A_PromptV4/apeSkim7.root',
       '/store/caf/user/hauk/data/DoubleMu/Run2011A_PromptV4/apeSkim8.root',
       '/store/caf/user/hauk/data/DoubleMu/Run2011A_PromptV4/apeSkim9.root',
       '/store/caf/user/hauk/data/DoubleMu/Run2011A_PromptV4/apeSkim10.root',

] )
