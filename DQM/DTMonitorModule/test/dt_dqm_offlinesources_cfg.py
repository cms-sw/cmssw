import FWCore.ParameterSet.Config as cms

process = cms.Process("DTDQMOfflineSources")

# the source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/data/CRAFT09/Cosmics/RAW/v1/000/109/459/F2C412D1-4D7D-DE11-8C60-000423D99E46.root',
        '/store/data/CRAFT09/Cosmics/RAW/v1/000/109/459/EAC40AC3-527D-DE11-A2BF-000423D952C0.root',
        '/store/data/CRAFT09/Cosmics/RAW/v1/000/109/459/EA44F625-4D7D-DE11-A259-000423D9A212.root',
        '/store/data/CRAFT09/Cosmics/RAW/v1/000/109/459/F6F17AD3-4D7D-DE11-8E8C-000423D99F3E.root',
        '/store/data/CRAFT09/Cosmics/RAW/v1/000/109/459/E6ACE824-4D7D-DE11-93CF-000423D996C8.root',
        '/store/data/CRAFT09/Cosmics/RAW/v1/000/109/459/DA838A3D-4F7D-DE11-AA12-001D09F248F8.root',
      '/store/data/CRAFT09/Cosmics/RAW/v1/000/109/459/FA1FEED3-4D7D-DE11-A695-000423D98B6C.root'
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
    )


#process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("CondCore.DBCommon.CondDBSetup_cfi")


# Conditions (Global Tag is used here):

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cfi")
process.GlobalTag.globaltag = "GR09_31X_V5P::All"
#process.prefer("GlobalTag")

# Magnetic fiuld: force mag field to be 3.8 tesla
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

#Geometry
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")


# Real data raw to digi
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")

# reconstruction sequence for Cosmics
process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")
#process.load("RecoLocalMuon.Configuration.RecoLocalMuonCosmics_cff")
#process.load("RecoMuon.Configuration.RecoMuonCosmics_cff")



# offline DQM
process.load("DQM.DTMonitorModule.dtDQMOfflineSources_cff")
process.load("DQMServices.Components.MEtoEDMConverter_cff")



process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('drop *', 
                                                                      'keep *_MEtoEDMConverter_*_*'),
                               fileName = cms.untracked.string('DTDQMOffline.root')
                               )








# message logger
process.MessageLogger = cms.Service("MessageLogger",
                                    debugModules = cms.untracked.vstring('*'),
                                    destinations = cms.untracked.vstring('cout'),
                                    categories = cms.untracked.vstring('DTChamberEfficiency'), 
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING'),
                                                              noLineBreaks = cms.untracked.bool(False),
                                                              DEBUG = cms.untracked.PSet(
                                                                      limit = cms.untracked.int32(0)),
                                                              INFO = cms.untracked.PSet(
                                                                      limit = cms.untracked.int32(0)),
                                                              DTChamberEfficiency = cms.untracked.PSet(
                                                                                 limit = cms.untracked.int32(-1))
                                                              )
                                    )




# raw to digi
process.unpackers = cms.Sequence(process.gtDigis + process.dttfDigis + process.muonDTDigis + process.muonCSCDigis + process.muonRPCDigis)
# reco
#process.reco = cms.Sequence(process.dt1DRecHits + process.dt4DSegments + process.muonRecoGR)
process.reco = cms.Sequence(process.offlineBeamSpot + process.muonsLocalRecoCosmics + process.STAmuontrackingforcosmics)

process.DTDQMOfflineCosmics = cms.Sequence(process.dtSources)


#Paths
process.allPath = cms.Path(process.unpackers *
                           process.reco *
                           process.DTDQMOfflineCosmics *
                           process.MEtoEDMConverter)


process.outpath = cms.EndPath(process.out)




# f = file('aNewconfigurationFile.cfg', 'w')
# f.write(process.dumpConfig())
# f.close()


