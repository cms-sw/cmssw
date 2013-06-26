import FWCore.ParameterSet.Config as cms

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)

readFiles.extend( [
       '/store/relval/CMSSW_3_1_0/RelValZEE/GEN-SIM-RECO/MC_31X_V1-v1/0001/E0B5B4FD-C066-DE11-AB1E-001D09F24600.root',
       '/store/relval/CMSSW_3_1_0/RelValZEE/GEN-SIM-RECO/MC_31X_V1-v1/0001/B0E8F011-C366-DE11-A49F-001D09F2516D.root',
       '/store/relval/CMSSW_3_1_0/RelValZEE/GEN-SIM-RECO/MC_31X_V1-v1/0001/6A7A4717-C166-DE11-83FF-001D09F24FEC.root',
       '/store/relval/CMSSW_3_1_0/RelValZEE/GEN-SIM-RECO/MC_31X_V1-v1/0001/58E1635B-DE66-DE11-9CDD-0019B9F6C674.root' ] );


secFiles.extend( [
               ] )

