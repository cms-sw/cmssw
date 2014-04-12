import FWCore.ParameterSet.Config as cms

readFiles = cms.untracked.vstring()
source = cms.Source ("PoolSource",fileNames = readFiles)
readFiles.extend( ( 
          " /store/relval/CMSSW_3_4_0_pre2/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_3XY_V10-v1/0003/D085615A-A5BD-DE11-8897-0026189437E8.root",
          "/store/relval/CMSSW_3_4_0_pre2/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_3XY_V10-v1/0003/9A67BC35-AFBD-DE11-9FE5-001731AF67B5.root",
          "/store/relval/CMSSW_3_4_0_pre2/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_3XY_V10-v1/0003/5C70929C-C0BD-DE11-B1C6-002618943829.root",
          "/store/relval/CMSSW_3_4_0_pre2/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_3XY_V10-v1/0003/54E06430-B1BD-DE11-BC8C-003048679168.root",
          "/store/relval/CMSSW_3_4_0_pre2/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_3XY_V10-v1/0003/32097BFD-A3BD-DE11-894F-001A92810ADE.root"
      ) 
)
