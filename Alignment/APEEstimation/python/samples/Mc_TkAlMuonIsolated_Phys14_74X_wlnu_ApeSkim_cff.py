import FWCore.ParameterSet.Config as cms

maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
      '/store/caf/user/cschomak/wlnu/wlnu1.root',
      '/store/caf/user/cschomak/wlnu/wlnu2.root',
      '/store/caf/user/cschomak/wlnu/wlnu3.root',
      '/store/caf/user/cschomak/wlnu/wlnu4.root',
      '/store/caf/user/cschomak/wlnu/wlnu5.root',
      '/store/caf/user/cschomak/wlnu/wlnu6.root',
      '/store/caf/user/cschomak/wlnu/wlnu7.root',
      '/store/caf/user/cschomak/wlnu/wlnu8.root',
      '/store/caf/user/cschomak/wlnu/wlnu9.root',
       
       ]);


secFiles.extend( [
               ] )


