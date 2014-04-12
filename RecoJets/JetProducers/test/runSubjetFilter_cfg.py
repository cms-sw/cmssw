import FWCore.ParameterSet.Config as cms


#!
#! PROCESS
#!
process = cms.Process("SJF")



#!
#! INPUT
#!
qcdFiles = cms.untracked.vstring(
    '/store/relval/CMSSW_3_3_1/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V9-v3/0003/FC5633F4-CAC0-DE11-9B8C-0030487C6090.root',
    '/store/relval/CMSSW_3_3_1/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V9-v3/0003/AC1E71D1-C9C0-DE11-AD08-0030487C6090.root',
    '/store/relval/CMSSW_3_3_1/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V9-v3/0003/AAEB831F-E4C0-DE11-85CC-0030487C6090.root',
    '/store/relval/CMSSW_3_3_1/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V9-v3/0003/80B9B076-C6C0-DE11-970A-000423D98B6C.root',
    '/store/relval/CMSSW_3_3_1/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V9-v3/0003/7A6512C0-9CC1-DE11-9228-0030487A18F2.root',
    '/store/relval/CMSSW_3_3_1/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V9-v3/0003/1CE3F169-C7C0-DE11-914D-0030487C6090.root',
    '/store/relval/CMSSW_3_3_1/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V9-v3/0003/16D5C17F-C8C0-DE11-9BEE-0030487A1990.root'    
    )

ttbarFiles = cms.untracked.vstring(
    '/store/relval/CMSSW_3_3_0/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v2/0003/8C293B1C-7DBD-DE11-A138-002618943969.root',
    '/store/relval/CMSSW_3_3_0/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v2/0002/80746BC6-E1BC-DE11-B6AF-0026189438F4.root',
    '/store/relval/CMSSW_3_3_0/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v2/0002/76B9924A-E2BC-DE11-8D8F-001A92971B06.root',
    '/store/relval/CMSSW_3_3_0/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v2/0002/6A476ECE-E7BC-DE11-941A-002618943923.root',
    '/store/relval/CMSSW_3_3_0/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v2/0002/585171F0-4CBD-DE11-B96D-001A92971AA4.root',
    '/store/relval/CMSSW_3_3_0/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v2/0002/348829ED-E9BC-DE11-B2A5-0026189438C4.root',
    '/store/relval/CMSSW_3_3_0/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v2/0002/249CA932-E9BC-DE11-9C34-00261894393E.root'
    )

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))
process.source = cms.Source("PoolSource", fileNames = ttbarFiles)


#!
#! SERVICES
#!
process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    cout         = cms.untracked.PSet(threshold = cms.untracked.string('WARNING'))
)


#!
#! JET RECONSTRUCTION
#!
process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('RecoJets.Configuration.GenJetParticles_cff')
process.load('RecoJets.JetProducers.caSubjetFilterGenJets_cfi')
process.load('RecoJets.JetProducers.caSubjetFilterCaloJets_cfi')
process.load('RecoJets.JetProducers.caSubjetFilterPFJets_cfi')


#!
#! RUN
#!
process.run = cms.Path(
    process.genParticlesForJets*
    process.caSubjetFilterGenJets*
    process.caSubjetFilterCaloJets*
    process.caSubjetFilterPFJets
    )

