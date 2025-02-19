import FWCore.ParameterSet.Config as cms

process = cms.Process("OWNPARTICLES")

process.load("FWCore.MessageService.MessageLogger_cfi")
## configure geometry
process.load("Configuration.StandardSequences.Geometry_cff")
## configure B field
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff")
process.GlobalTag.globaltag = "START3X_V20::All"

process.load("RecoMET.METAlgorithms.MuonMETValueMapProducer_cff")
process.load("RecoMET.METAlgorithms.MetMuonCorrections_cff")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = ''
process.MessageLogger.cerr.FwkReport.reportEvery = 100
#use 900 GeV JEC

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(5000) )

#process.source = cms.Source("PoolSource",
#    # replace 'myfile.root' with the source file you want to use
#    fileNames = cms.untracked.vstring(
#    '/store/relval/CMSSW_3_5_0_pre5/RelValWM/GEN-SIM-RECO/START3X_V20-v2/0009/FE457857-240F-DF11-8AAE-0030487A195C.root',
#    '/store/relval/CMSSW_3_5_0_pre5/RelValWM/GEN-SIM-RECO/START3X_V20-v2/0009/F22BF489-230F-DF11-9815-0030487CD6DA.root',
#    '/store/relval/CMSSW_3_5_0_pre5/RelValWM/GEN-SIM-RECO/START3X_V20-v2/0009/E42770F9-260F-DF11-A482-0030487C60AE.root',
#    '/store/relval/CMSSW_3_5_0_pre5/RelValWM/GEN-SIM-RECO/START3X_V20-v2/0009/3AEF3A5C-270F-DF11-949B-0030487C7828.root',
#    '/store/relval/CMSSW_3_5_0_pre5/RelValWM/GEN-SIM-RECO/START3X_V20-v2/0009/00268418-C50F-DF11-B801-000423D98930.root')
#)
process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_5_0_pre5/RelValInclusiveppMuX/GEN-SIM-RECO/MC_3XY_V20-v1/0011/1239FA32-100F-DF11-8F13-003048678B0C.root',
    '/store/relval/CMSSW_3_5_0_pre5/RelValInclusiveppMuX/GEN-SIM-RECO/MC_3XY_V20-v1/0010/D08819B4-C10E-DF11-8349-00261894393B.root',
    '/store/relval/CMSSW_3_5_0_pre5/RelValInclusiveppMuX/GEN-SIM-RECO/MC_3XY_V20-v1/0010/BEE58FCF-CB0E-DF11-95E6-0018F3D09686.root',
    '/store/relval/CMSSW_3_5_0_pre5/RelValInclusiveppMuX/GEN-SIM-RECO/MC_3XY_V20-v1/0010/B6DD40D2-CD0E-DF11-9B40-0018F3D09620.root',
    '/store/relval/CMSSW_3_5_0_pre5/RelValInclusiveppMuX/GEN-SIM-RECO/MC_3XY_V20-v1/0010/B0937D45-C30E-DF11-AF2A-001A92971B94.root',
   '/store/relval/CMSSW_3_5_0_pre5/RelValInclusiveppMuX/GEN-SIM-RECO/MC_3XY_V20-v1/0010/ACDB5440-CB0E-DF11-BDEC-001A9281170E.root',
    '/store/relval/CMSSW_3_5_0_pre5/RelValInclusiveppMuX/GEN-SIM-RECO/MC_3XY_V20-v1/0010/76F8561C-CA0E-DF11-8333-001A92811702.root',
    '/store/relval/CMSSW_3_5_0_pre5/RelValInclusiveppMuX/GEN-SIM-RECO/MC_3XY_V20-v1/0010/745794AE-CA0E-DF11-A621-0018F3D09626.root',
    '/store/relval/CMSSW_3_5_0_pre5/RelValInclusiveppMuX/GEN-SIM-RECO/MC_3XY_V20-v1/0010/74282050-CC0E-DF11-85D1-001A92971AD8.root',
    '/store/relval/CMSSW_3_5_0_pre5/RelValInclusiveppMuX/GEN-SIM-RECO/MC_3XY_V20-v1/0010/6AD7B8F0-CF0E-DF11-8655-0018F3D096C6.root',
    '/store/relval/CMSSW_3_5_0_pre5/RelValInclusiveppMuX/GEN-SIM-RECO/MC_3XY_V20-v1/0010/5CEFCFEB-CC0E-DF11-A4D3-001A928116C4.root',
    '/store/relval/CMSSW_3_5_0_pre5/RelValInclusiveppMuX/GEN-SIM-RECO/MC_3XY_V20-v1/0010/4AC1BBF6-C50E-DF11-ACF1-001A92811722.root',
    '/store/relval/CMSSW_3_5_0_pre5/RelValInclusiveppMuX/GEN-SIM-RECO/MC_3XY_V20-v1/0010/2EF33D1A-C90E-DF11-80BA-003048678B1C.root',
    '/store/relval/CMSSW_3_5_0_pre5/RelValInclusiveppMuX/GEN-SIM-RECO/MC_3XY_V20-v1/0010/2A9DF3D9-CB0E-DF11-8D57-00304867C1B0.root',
    '/store/relval/CMSSW_3_5_0_pre5/RelValInclusiveppMuX/GEN-SIM-RECO/MC_3XY_V20-v1/0010/20CC65A5-C80E-DF11-81BD-001A92810AF2.root')
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('myOutputFileAll_MuX.root')
)

process.out.outputCommands = cms.untracked.vstring( 'drop *' )
process.out.outputCommands.extend(cms.untracked.vstring('keep recoCalo*_*_*_*'))
process.out.outputCommands.extend(cms.untracked.vstring('keep *_*_*_OWNPARTICLES'))
process.out.outputCommands.extend(cms.untracked.vstring('keep recoMuons_muons__RECO'))
process.out.outputCommands.extend(cms.untracked.vstring('keep *MET*_*_*_*'))
process.out.outputCommands.extend(cms.untracked.vstring('keep *_*met*_*_*'))
process.out.outputCommands.extend(cms.untracked.vstring('keep *_*Met*_*_*'))


  
process.p = cms.Path(process.muonMETValueMapProducer*process.corMetGlobalMuons)

process.e = cms.EndPath(process.out)
