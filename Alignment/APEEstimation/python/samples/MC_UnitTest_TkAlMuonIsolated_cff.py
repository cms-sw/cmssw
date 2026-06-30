import FWCore.ParameterSet.Config as cms

maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)


readFiles.extend( [
    "/store/relval/CMSSW_20_0_0_pre1/RelValSingleMuPt1000/ALCARECO/TkAlMuonIsolated-150X_mcRun4_realistic_v1_STD_RegeneratedGS_D121_noPU-v1/2590000/f71df48e-af79-4e74-ab94-352f508449dd.root"] );

secFiles.extend( [
               ] )

