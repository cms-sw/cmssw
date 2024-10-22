import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
       '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0000/FE32B1E4-C7FA-DD11-A2FD-001A92971ADC.root'),
                            secondaryFileNames = cms.untracked.vstring(
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/000/708C5612-CFA5-DD11-AD52-0019DB29C5FC.root',
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/000/38419E41-D1A5-DD11-8B68-001617C3B6E2.root',
       '/store/data/Commissioning08/Cosmics/RAW/v1/000/068/000/2CDF3B0F-CFA5-DD11-AE18-000423D99A8E.root')
 )


process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/DPGAnalysis/Skims/python/TrackerPointing_ToscaMap090322_cfg.py,v $'),
    annotation = cms.untracked.string('CRAFT TrackerPointing skim')
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration.StandardSequences.GeometryDB_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'auto:run3_data_prompt'
process.prefer("GlobalTag")

process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")




process.cosmictrackfinderP5TkCntFilter = cms.EDFilter("TrackCountFilter",
                                                      src = cms.InputTag('cosmictrackfinderP5'),
                                                      minNumber = cms.uint32(1) 
                                                      )

process.ctfWithMaterialTracksP5TkCntFilter = cms.EDFilter("TrackCountFilter",
                                                          src = cms.InputTag('ctfWithMaterialTracksP5'),
                                                          minNumber = cms.uint32(1) 
                                                          )

process.rsWithMaterialTracksP5TkCntFilter = cms.EDFilter("TrackCountFilter",
                                                         src = cms.InputTag('rsWithMaterialTracksP5'),
                                                         minNumber = cms.uint32(1) 
                                                         )

process.cosmicMuonsBarrelOnlyTkFilter = cms.EDFilter("HLTMuonPointingFilter",
                                                     SALabel = cms.InputTag("cosmicMuonsBarrelOnly"),
                                                     PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                                     radius = cms.double(90.0),
                                                     maxZ = cms.double(130.0)
                                                     )

process.cosmicMuonsBarrelOnlyTkPath = cms.Path(process.cosmicMuonsBarrelOnlyTkFilter)
process.cosmictrackfinderP5TkCntPath = cms.Path(process.cosmictrackfinderP5TkCntFilter)
process.ctfWithMaterialTracksP5TkCntPath = cms.Path(process.ctfWithMaterialTracksP5TkCntFilter)
process.rsWithMaterialTracksP5TkCntPath = cms.Path(process.rsWithMaterialTracksP5TkCntFilter)


process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('keep *','drop *_MEtoEDMConverter_*_*'),
                               SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('cosmicMuonsBarrelOnlyTkPath',
                                                                                            'cosmictrackfinderP5TkCntPath',
                                                                                            'ctfWithMaterialTracksP5TkCntPath',
                                                                                            'rsWithMaterialTracksP5TkCntPath')),
                               dataset = cms.untracked.PSet(
			                 dataTier = cms.untracked.string('GEN-SIM-RAW-RECO'),
                                         filterName = cms.untracked.string('TrackingPointing')),
                               fileName = cms.untracked.string('trackerPointing.root')
                               )

process.this_is_the_end = cms.EndPath(process.out)
