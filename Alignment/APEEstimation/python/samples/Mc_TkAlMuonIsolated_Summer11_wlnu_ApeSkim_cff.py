import FWCore.ParameterSet.Config as cms

maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)


readFiles.extend( [

       '/store/caf/user/hauk/mc/Summer11/wlnu/apeSkim1.root',
       '/store/caf/user/hauk/mc/Summer11/wlnu/apeSkim2.root',
       '/store/caf/user/hauk/mc/Summer11/wlnu/apeSkim3.root',
       '/store/caf/user/hauk/mc/Summer11/wlnu/apeSkim4.root',
       '/store/caf/user/hauk/mc/Summer11/wlnu/apeSkim5.root',
       '/store/caf/user/hauk/mc/Summer11/wlnu/apeSkim6.root',
       '/store/caf/user/hauk/mc/Summer11/wlnu/apeSkim7.root',
       '/store/caf/user/hauk/mc/Summer11/wlnu/apeSkim8.root',
       '/store/caf/user/hauk/mc/Summer11/wlnu/apeSkim9.root',
       '/store/caf/user/hauk/mc/Summer11/wlnu/apeSkim10.root',
       '/store/caf/user/hauk/mc/Summer11/wlnu/apeSkim11.root',
       '/store/caf/user/hauk/mc/Summer11/wlnu/apeSkim12.root',
       '/store/caf/user/hauk/mc/Summer11/wlnu/apeSkim13.root',
       '/store/caf/user/hauk/mc/Summer11/wlnu/apeSkim14.root',
       '/store/caf/user/hauk/mc/Summer11/wlnu/apeSkim15.root',

] )
