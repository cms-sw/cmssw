import FWCore.ParameterSet.Config as cms
 
process = cms.Process("Extrapolation")

### standard MessageLoggerConfiguration
process.load("FWCore.MessageService.MessageLogger_cfi")

### Standard Configurations
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR10_P_V8::All'


### Extrapolation
process.load("RecoJets.JetAssociationProducers.trackExtrapolator_cfi")

process.maxEvents = cms.untracked.PSet(     input = cms.untracked.int32(-1)     
)

process.source = cms.Source("PoolSource",
### tracks from collisions                            
    fileNames = cms.untracked.vstring(
'/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/BSCNOBEAMHALO-Feb9Skim_v1/0030/CE75A97D-AF18-DF11-ACA1-002618943866.root'
) 
) 

process.TRACKS = cms.OutputModule("PoolOutputModule",
                                outputCommands = cms.untracked.vstring('drop *_*_*_*', 
                                                                       'keep recoTracks_*_*_*',
                                                                       'keep recoTrackExtras_*_*_*',
                                                                       'keep TrackingRecHitsOwned_*_*_*',
                                                                       'keep *_trackExtrapolator_*_*'),

                                fileName = cms.untracked.string('extrapolation.root')
                                )

process.options = cms.untracked.PSet(     wantSummary = cms.untracked.bool(True) )


process.p1 = cms.Path(process.trackExtrapolator
)
process.outpath = cms.EndPath(process.TRACKS)

