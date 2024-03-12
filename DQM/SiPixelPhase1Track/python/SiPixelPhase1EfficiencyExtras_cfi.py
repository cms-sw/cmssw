import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester


SiPixelPhase1EfficiencyExtras = DQMEDHarvester("SiPixelPhase1EfficiencyExtras",
                                     EffFolderName = cms.string('PixelPhase1/Tracks/'),
                                     VtxFolderName = cms.string('Tracking/TrackParameters/generalTracks/GeneralProperties/'),
				     InstLumiFolderName = cms.string('HLT/LumiMonitoring/')
)
# foo bar baz
# Qv2aZR8RLNPTc
# G0uKcVqBAsOr0
