import FWCore.ParameterSet.Config as cms

process = cms.Process("DTDQMOfflineSources")

# the source
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
    '/store/data/CRAFT09/Cosmics/RECO/v1/000/109/459/F8EA0C0B-8E7D-DE11-A114-001D09F23A6B.root',
    '/store/data/CRAFT09/Cosmics/RECO/v1/000/109/459/F669FC3F-977D-DE11-BFB6-001D09F241B9.root',
    '/store/data/CRAFT09/Cosmics/RECO/v1/000/109/459/EEA05855-927D-DE11-BDBC-000423D99B3E.root',
    '/store/data/CRAFT09/Cosmics/RECO/v1/000/109/459/EC6E901F-A17D-DE11-8DC1-0019B9F72F97.root',
    '/store/data/CRAFT09/Cosmics/RECO/v1/000/109/459/EA01D71C-957D-DE11-B32D-001D09F2545B.root',
    '/store/data/CRAFT09/Cosmics/RECO/v1/000/109/459/D6772604-937D-DE11-B191-000423D98834.root',
    '/store/data/CRAFT09/Cosmics/RECO/v1/000/109/459/D4EEE43A-977D-DE11-90C3-001D09F2AD7F.root'
    ),
                            inputCommands = cms.untracked.vstring('keep *', 'drop *_MEtoEDMConverter_*_*')                
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
    )


#process.load("FWCore.MessageLogger.MessageLogger_cfi")

from RecoMuon.TrackingTools.MuonServiceProxy_cff import * 
from RecoMuon.TrackingTools.MuonTrackLoader_cff import *

process.load("CondCore.DBCommon.CondDBSetup_cfi")

# Conditions (Global Tag is used here):
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = "CRAFT_V4P::All"
process.GlobalTag.globaltag = "GR09_31X_V5P::All"
#process.prefer("GlobalTag")

# Magnetic fiuld: force mag field to be 3.8 tesla
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

#Geometry
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

# Real data raw to digi
# process.load("Configuration.StandardSequences.RawToDigi_Data_cff")

# reconstruction sequence for Cosmics
process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")
#process.load("RecoLocalMuon.Configuration.RecoLocalMuonCosmics_cff")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

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
                                    categories = cms.untracked.vstring('DTTimeEvolutionHisto'), 
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING'),
                                                              noLineBreaks = cms.untracked.bool(False),
                                                              DEBUG = cms.untracked.PSet(
                                                                      limit = cms.untracked.int32(0)),
                                                              INFO = cms.untracked.PSet(
                                                                      limit = cms.untracked.int32(0)),
                                                              DTTimeEvolutionHisto = cms.untracked.PSet(limit = cms.untracked.int32(-1))
                                                              )
                                    )


# raw to digi
# process.unpackers = cms.Sequence(process.muonDTDigis)
# reco
# process.reco = cms.Sequence(process.dt1DRecHits + process.dt4DSegments)

process.dtSources.remove(process.dtDataIntegrityUnpacker)
process.DTDQMOfflineCosmics = cms.Sequence(process.dtSources)



#Paths
process.allPath = cms.Path(process.DTDQMOfflineCosmics *
                           process.MEtoEDMConverter)


process.outpath = cms.EndPath(process.out)




# f = file('aNewconfigurationFile.cfg', 'w')
# f.write(process.dumpConfig())
# f.close()


