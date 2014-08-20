import FWCore.ParameterSet.Config as cms

maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10000) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [ 'file:/afs/cern.ch/user/l/lgray/work/public/CMSSW_7_2_0_pre4/src/miniAOD-prod_PAT.root'
        ] );


secFiles.extend( [
               ] )

