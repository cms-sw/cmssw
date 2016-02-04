import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContentCosmics_cff import FEVTEventContent
skimContent = FEVTEventContent.clone()
skimContent.outputCommands.append("drop *_MEtoEDMConverter_*_*")
skimContent.outputCommands.append("drop *_*_*_SKIM")

from DPGAnalysis.Skims.cosmicSPSkim_cff import *

cosmicMuonsBarrelOnlyPath = cms.Path(cosmicMuonsBarrelOnlySequence)
cosmicMuonsPath = cms.Path(cosmicMuonsSequence)
cosmicMuons1LegPath = cms.Path(cosmicMuons1LegSequence)
globalCosmicMuonsBarrelOnlyPath = cms.Path(globalCosmicMuonsBarrelOnlySequence)
cosmictrackfinderP5Path = cms.Path(cosmictrackfinderP5Sequence)
globalCosmicMuonsPath = cms.Path(globalCosmicMuonsSequence)
rsWithMaterialTracksP5Path = cms.Path(rsWithMaterialTracksP5Sequence)
globalCosmicMuons1LegPath = cms.Path(globalCosmicMuons1LegSequence)
rsWithMaterialTracksP5Path = cms.Path(rsWithMaterialTracksP5Sequence)

SKIMStreamCosmicSP = cms.FilteredStream(
            responsible = '',
                    name = 'CosmicSP',
                    paths = (cosmicMuonsBarrelOnlyPath,
                                              cosmicMuonsPath,
                                              cosmicMuons1LegPath,
                                              globalCosmicMuonsBarrelOnlyPath,
                                              cosmictrackfinderP5Path,
                                              globalCosmicMuonsPath,
                                              rsWithMaterialTracksP5Path,
                                              globalCosmicMuons1LegPath,
                                              rsWithMaterialTracksP5Path),
                    content = skimContent.outputCommands,
                    selectEvents = cms.untracked.PSet(),
                    dataTier = cms.untracked.string('RAW-RECO')
                    )

#####################

from DPGAnalysis.Skims.cosmicTPSkim_cff import *

cosmicMuonsBarrelOnlyTkPath = cms.Path(cosmicMuonsEndCapsOnlyTkSequence)
cosmicMuonsEndCapsOnlyTkPath = cms.Path(cosmicMuonsTkSequence)
cosmicMuonsTkPath = cms.Path(cosmicMuons1LegTkSequence)
cosmicMuons1LegTkPath = cms.Path(cosmicMuons1LegTkSequence)
globalCosmicMuonsBarrelOnlyTkPath = cms.Path(globalCosmicMuonsBarrelOnlyTkSequence)
globalCosmicMuonsEndCapsOnlyTkPath = cms.Path(globalCosmicMuonsEndCapsOnlyTkSequence)
globalCosmicMuonsTkPath = cms.Path(globalCosmicMuonsTkSequence)
globalCosmicMuons1LegTkPath = cms.Path(globalCosmicMuons1LegTkSequence)
cosmictrackfinderP5TkCntPath = cms.Path(cosmictrackfinderP5TkCntSequence)
ctfWithMaterialTracksP5TkCntPath = cms.Path(ctfWithMaterialTracksP5TkCntSequence)
rsWithMaterialTracksP5TkCntPath = cms.Path(rsWithMaterialTracksP5TkCntSequence)

SKIMStreamCosmicTP = cms.FilteredStream(
            responsible = '',
                    name = 'CosmicTP',
                    paths = (cosmicMuonsBarrelOnlyTkPath,
                                              cosmicMuonsEndCapsOnlyTkPath,
                                              cosmicMuonsTkPath,
                                              cosmicMuons1LegTkPath,
                                              globalCosmicMuonsBarrelOnlyTkPath,
                                              globalCosmicMuonsEndCapsOnlyTkPath,
                                              globalCosmicMuonsTkPath,
                                              globalCosmicMuons1LegTkPath,
                                              cosmictrackfinderP5TkCntPath,
                                              ctfWithMaterialTracksP5TkCntPath,
                                              rsWithMaterialTracksP5TkCntPath),
                    content = skimContent.outputCommands,
                    selectEvents = cms.untracked.PSet(),
                    dataTier = cms.untracked.string('RAW-RECO')
                    )

#####################

