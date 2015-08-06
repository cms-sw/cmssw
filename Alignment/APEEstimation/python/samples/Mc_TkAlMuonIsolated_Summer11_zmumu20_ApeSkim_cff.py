import FWCore.ParameterSet.Config as cms

maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)


readFiles.extend( [

       '/store/caf/user/hauk/mc/Summer11/zmumu20/apeSkim1.root',
       '/store/caf/user/hauk/mc/Summer11/zmumu20/apeSkim2.root',
       '/store/caf/user/hauk/mc/Summer11/zmumu20/apeSkim3.root',
       '/store/caf/user/hauk/mc/Summer11/zmumu20/apeSkim4.root',
       '/store/caf/user/hauk/mc/Summer11/zmumu20/apeSkim5.root',
       '/store/caf/user/hauk/mc/Summer11/zmumu20/apeSkim6.root',
       '/store/caf/user/hauk/mc/Summer11/zmumu20/apeSkim7.root',
       '/store/caf/user/hauk/mc/Summer11/zmumu20/apeSkim8.root',
       '/store/caf/user/hauk/mc/Summer11/zmumu20/apeSkim9.root',
       '/store/caf/user/hauk/mc/Summer11/zmumu20/apeSkim10.root',
       '/store/caf/user/hauk/mc/Summer11/zmumu20/apeSkim11.root',
       '/store/caf/user/hauk/mc/Summer11/zmumu20/apeSkim12.root',
       '/store/caf/user/hauk/mc/Summer11/zmumu20/apeSkim13.root',
       '/store/caf/user/hauk/mc/Summer11/zmumu20/apeSkim14.root',
       '/store/caf/user/hauk/mc/Summer11/zmumu20/apeSkim15.root',
       '/store/caf/user/hauk/mc/Summer11/zmumu20/apeSkim16.root',
       '/store/caf/user/hauk/mc/Summer11/zmumu20/apeSkim17.root',
       '/store/caf/user/hauk/mc/Summer11/zmumu20/apeSkim18.root',
       '/store/caf/user/hauk/mc/Summer11/zmumu20/apeSkim19.root',
       '/store/caf/user/hauk/mc/Summer11/zmumu20/apeSkim20.root',
       '/store/caf/user/hauk/mc/Summer11/zmumu20/apeSkim21.root',

] )
