import FWCore.ParameterSet.Config as cms

maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
       '/store/relval/CMSSW_3_1_0/RelValZMM/GEN-SIM-RECO/STARTUP31X_V1-v1/0001/D22D3E9C-8966-DE11-900A-001617C3B66C.root',
       '/store/relval/CMSSW_3_1_0/RelValZMM/GEN-SIM-RECO/STARTUP31X_V1-v1/0001/7E8944E8-8E66-DE11-9BBF-001D09F23A84.root',
       '/store/relval/CMSSW_3_1_0/RelValZMM/GEN-SIM-RECO/STARTUP31X_V1-v1/0001/384836AE-D166-DE11-8D68-001D09F2983F.root',
       '/store/relval/CMSSW_3_1_0/RelValZMM/GEN-SIM-RECO/STARTUP31X_V1-v1/0001/04E84AF4-8366-DE11-BC25-001D09F28D4A.root' ] );


secFiles.extend( [
               ] )

