import FWCore.ParameterSet.Config as cms

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
       '/store/relval/CMSSW_3_1_4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/MC_31X_V3_FastSim-v1/0005/FC3E812E-5CB0-DE11-BBAB-001D09F290CE.root',
       '/store/relval/CMSSW_3_1_4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/MC_31X_V3_FastSim-v1/0005/DAE98CF4-5BB0-DE11-B5F9-001D09F28EC1.root',
       '/store/relval/CMSSW_3_1_4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/MC_31X_V3_FastSim-v1/0005/D67AB8E1-5BB0-DE11-9958-000423D94494.root',
       '/store/relval/CMSSW_3_1_4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/MC_31X_V3_FastSim-v1/0005/C278290E-5CB0-DE11-8B43-001D09F24FBA.root',
       '/store/relval/CMSSW_3_1_4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/MC_31X_V3_FastSim-v1/0005/BC049595-5BB0-DE11-9001-001D09F28755.root',
       '/store/relval/CMSSW_3_1_4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/MC_31X_V3_FastSim-v1/0005/B474BFDA-5BB0-DE11-BCFA-001D09F295A1.root',
       '/store/relval/CMSSW_3_1_4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/MC_31X_V3_FastSim-v1/0005/A6EDA51A-5CB0-DE11-8E99-0019B9F72CE5.root',
       '/store/relval/CMSSW_3_1_4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/MC_31X_V3_FastSim-v1/0005/A498ABDF-5BB0-DE11-A5A4-001D09F28F25.root',
       '/store/relval/CMSSW_3_1_4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/MC_31X_V3_FastSim-v1/0005/942F3ADE-5BB0-DE11-869D-000423D991D4.root',
       '/store/relval/CMSSW_3_1_4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/MC_31X_V3_FastSim-v1/0005/88CED144-5CB0-DE11-929C-000423D94990.root',
       '/store/relval/CMSSW_3_1_4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/MC_31X_V3_FastSim-v1/0005/863DD28C-5BB0-DE11-A108-001D09F24FBA.root',
       '/store/relval/CMSSW_3_1_4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/MC_31X_V3_FastSim-v1/0005/809E8325-5CB0-DE11-9BCD-000423D99B3E.root',
       '/store/relval/CMSSW_3_1_4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/MC_31X_V3_FastSim-v1/0005/02929F11-5CB0-DE11-A77D-000423D94494.root',
       '/store/relval/CMSSW_3_1_4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/MC_31X_V3_FastSim-v1/0005/022203F1-5BB0-DE11-9643-001D09F2AD7F.root',
       '/store/relval/CMSSW_3_1_4/RelValQCD_FlatPt_15_3000/GEN-SIM-DIGI-RECO/MC_31X_V3_FastSim-v1/0005/00F53826-5CB0-DE11-9FBB-000423D991D4.root' ] );


secFiles.extend( [
               ] )

