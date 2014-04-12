import FWCore.ParameterSet.Config as cms

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [

    '/store/relval/CMSSW_3_8_5/RelValZMM/GEN-SIM-RECO/START38_V12-v1/0041/1C1BBE0B-D2D2-DF11-BDA3-002618943852.root',
    '/store/relval/CMSSW_3_8_5/RelValZMM/GEN-SIM-RECO/START38_V12-v1/0040/7EC3ACC4-E9D1-DF11-B4F3-0030486792B4.root',
    '/store/relval/CMSSW_3_8_5/RelValZMM/GEN-SIM-RECO/START38_V12-v1/0039/CA730041-E1D1-DF11-A9A0-0018F3D09620.root',
    '/store/relval/CMSSW_3_8_5/RelValZMM/GEN-SIM-RECO/START38_V12-v1/0039/A40E6B9A-E4D1-DF11-A8DD-002618943927.root',
    '/store/relval/CMSSW_3_8_5/RelValZMM/GEN-SIM-RECO/START38_V12-v1/0039/A06D1DB1-D9D1-DF11-9CE9-003048678FA0.root',
    '/store/relval/CMSSW_3_8_5/RelValZMM/GEN-SIM-RECO/START38_V12-v1/0039/14A658AC-D8D1-DF11-899B-002618FDA21D.root'
] );


secFiles.extend( [
               ] )

