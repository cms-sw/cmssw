import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/D266D139-D871-DE11-A709-001D09F28F0C.root',
'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/CA27788D-E871-DE11-8B46-001D09F276CF.root',
'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/AC5633B2-D471-DE11-9B3A-001D09F252F3.root',
'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/9CD957E7-D071-DE11-B6AE-001D09F252F3.root',
'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/94BF68F7-D171-DE11-902B-000423D986A8.root',
'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/7838FE1E-C771-DE11-9FD5-000423D98950.root',
'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/56632803-DD71-DE11-BAF5-000423D9870C.root',
'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/42A67CB9-E971-DE11-AA86-001D09F252F3.root',
'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/407225D3-D071-DE11-809B-001D09F297EF.root',
'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/3E5E1CF0-D271-DE11-AC2B-000423D94700.root',
'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/2C57E916-D071-DE11-AF0E-001D09F24E39.root',
'/store/data/Commissioning09/Cosmics/RECO/v5/000/105/755/228896A5-E571-DE11-A60B-001D09F2AF96.root'),

#  '/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v1/0002/0A12CE23-D7F9-DD11-819E-00E081348D21.root'),
                            secondaryFileNames = cms.untracked.vstring(
'/store/data/Commissioning09/Cosmics/RAW/v3/000/105/755/F6887FD0-9371-DE11-B69E-00304879FBB2.root',
'/store/data/Commissioning09/Cosmics/RAW/v3/000/105/755/EEAF292E-9571-DE11-9A17-000423D94C68.root',
'/store/data/Commissioning09/Cosmics/RAW/v3/000/105/755/ECC04DEB-9071-DE11-9F3A-001D09F23174.root',
'/store/data/Commissioning09/Cosmics/RAW/v3/000/105/755/E8CE8710-9171-DE11-9211-000423D94534.root',
'/store/data/Commissioning09/Cosmics/RAW/v3/000/105/755/DEB9057C-9471-DE11-BAF5-000423D94524.root',
'/store/data/Commissioning09/Cosmics/RAW/v3/000/105/755/DC2A7158-A171-DE11-ACD3-001D09F24047.root',
'/store/data/Commissioning09/Cosmics/RAW/v3/000/105/755/D6E68664-9271-DE11-AE97-000423D9970C.root',
'/store/data/Commissioning09/Cosmics/RAW/v3/000/105/755/D2D94C9C-9171-DE11-AA96-000423D94AA8.root',
'/store/data/Commissioning09/Cosmics/RAW/v3/000/105/755/B045C7CB-9371-DE11-AF39-001D09F24489.root',
'/store/data/Commissioning09/Cosmics/RAW/v3/000/105/755/A2C791EF-9071-DE11-871D-001D09F2423B.root',
'/store/data/Commissioning09/Cosmics/RAW/v3/000/105/755/9E1A4336-9071-DE11-81BD-001D09F251B8.root',
'/store/data/Commissioning09/Cosmics/RAW/v3/000/105/755/8C28F89C-9171-DE11-B944-000423D9A2AE.root',
'/store/data/Commissioning09/Cosmics/RAW/v3/000/105/755/82A37610-9371-DE11-A293-000423D98B6C.root',
'/store/data/Commissioning09/Cosmics/RAW/v3/000/105/755/822D1510-9371-DE11-A329-000423D99A8E.root',
'/store/data/Commissioning09/Cosmics/RAW/v3/000/105/755/76515611-9371-DE11-8BF2-001D09F24024.root',
'/store/data/Commissioning09/Cosmics/RAW/v3/000/105/755/76434EA1-9171-DE11-B39F-000423D98E54.root',
'/store/data/Commissioning09/Cosmics/RAW/v3/000/105/755/6E1DB87B-9471-DE11-B4E8-000423D98B28.root',
'/store/data/Commissioning09/Cosmics/RAW/v3/000/105/755/6A15A1C7-9371-DE11-9ACB-0030487A18F2.root',
'/store/data/Commissioning09/Cosmics/RAW/v3/000/105/755/4E95107B-9471-DE11-9E86-000423D944F0.root',
'/store/data/Commissioning09/Cosmics/RAW/v3/000/105/755/42C723CA-9871-DE11-8614-000423D6B48C.root',
'/store/data/Commissioning09/Cosmics/RAW/v3/000/105/755/3E934456-9271-DE11-BA19-000423D99AA2.root',
'/store/data/Commissioning09/Cosmics/RAW/v3/000/105/755/3AF89B38-9371-DE11-AEE9-000423D33970.root',
'/store/data/Commissioning09/Cosmics/RAW/v3/000/105/755/2C696C5B-9271-DE11-BD57-000423D94990.root')

#        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/578/085EFED4-E5AB-DD11-9ACA-001617C3B6FE.root')
)

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.14 $'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/DPGAnalysis/Skims/python/SuperPointing_cfg.py,v $'),
    annotation = cms.untracked.string('CRAFT SuperPointing skim')
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1000))
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))


process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration/StandardSequences/GeometryIdeal_cff')


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR09_31X_V3P::All' 

process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")


process.cosmicMuonsBarrelOnlyFilter = cms.EDFilter("HLTMuonPointingFilter",
                                                   SALabel = cms.string("cosmicMuonsBarrelOnly"),
                                                   PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                                   radius = cms.double(10.0),
                                                   maxZ = cms.double(50.0)
                                                   )

process.cosmicMuonsFilter = cms.EDFilter("HLTMuonPointingFilter",
                                         SALabel = cms.string("cosmicMuons"),
                                         PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                         radius = cms.double(10.0),
                                         maxZ = cms.double(50.0)
                                         )

process.cosmicMuons1LegFilter = cms.EDFilter("HLTMuonPointingFilter",
                                                       SALabel = cms.string("cosmicMuons1Leg"),
                                                       PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                                       radius = cms.double(10.0),
                                                       maxZ = cms.double(50.0)
                                                       )

process.globalCosmicMuonsBarrelOnlyFilter = cms.EDFilter("HLTMuonPointingFilter",
                                                         SALabel = cms.string("globalCosmicMuonsBarrelOnly"),
                                                         PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                                         radius = cms.double(10.0),
                                                         maxZ = cms.double(50.0)
                                                         )

process.cosmictrackfinderP5Filter = cms.EDFilter("HLTMuonPointingFilter",
                                                 SALabel = cms.string("cosmictrackfinderP5"),
                                                 PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                                 radius = cms.double(10.0),
                                                 maxZ = cms.double(50.0)
                                                 )

process.globalCosmicMuonsFilter = cms.EDFilter("HLTMuonPointingFilter",
                                               SALabel = cms.string("globalCosmicMuons"),
                                               PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                               radius = cms.double(10.0),
                                               maxZ = cms.double(50.0)
                                               )

process.rsWithMaterialTracksP5Filter = cms.EDFilter("HLTMuonPointingFilter",
                                                    SALabel = cms.string("rsWithMaterialTracksP5"),
                                                    PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                                    radius = cms.double(10.0),
                                                    maxZ = cms.double(50.0)
                                                    )

process.globalCosmicMuons1LegFilter = cms.EDFilter("HLTMuonPointingFilter",
                                                             SALabel = cms.string("globalCosmicMuons1Leg"),
                                                             PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                                             radius = cms.double(10.0),
                                                             maxZ = cms.double(50.0)
                                                             )

process.ctfWithMaterialTracksP5Filter = cms.EDFilter("HLTMuonPointingFilter",
                                                     SALabel = cms.string("ctfWithMaterialTracksP5"),
                                                     PropagatorName = cms.string("SteppingHelixPropagatorAny"),
                                                     radius = cms.double(10.0),
                                                     maxZ = cms.double(50.0)
                                                     )


process.cosmicMuonsBarrelOnlyPath = cms.Path(process.cosmicMuonsBarrelOnlyFilter)
process.cosmicMuonsPath = cms.Path(process.cosmicMuonsFilter)
process.cosmicMuons1LegPath = cms.Path(process.cosmicMuons1LegFilter)
process.globalCosmicMuonsBarrelOnlyPath = cms.Path(process.globalCosmicMuonsBarrelOnlyFilter)
process.cosmictrackfinderP5Path = cms.Path(process.cosmictrackfinderP5Filter)
process.globalCosmicMuonsPath = cms.Path(process.globalCosmicMuonsFilter)
process.rsWithMaterialTracksP5Path = cms.Path(process.rsWithMaterialTracksP5Filter)
process.globalCosmicMuons1LegPath = cms.Path(process.globalCosmicMuons1LegFilter)
process.ctfWithMaterialTracksP5Path = cms.Path(process.ctfWithMaterialTracksP5Filter)



process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('keep *','drop *_MEtoEDMConverter_*_*'),
                               SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('cosmicMuonsBarrelOnlyPath',
                                                                                            'cosmicMuonsPath',
                                                                                            'cosmicMuons1LegPath',
                                                                                            'globalCosmicMuonsBarrelOnlyPath',
                                                                                            'cosmictrackfinderP5Path',
                                                                                            'globalCosmicMuonsPath',
                                                                                            'rsWithMaterialTracksP5Path',
                                                                                            'globalCosmicMuons1LegPath',
                                                                                            'ctfWithMaterialTracksP5Path')),                               
                               dataset = cms.untracked.PSet(
			                 dataTier = cms.untracked.string('RAW-RECO'),
                                         filterName = cms.untracked.string('SuperPointing')),
                               fileName = cms.untracked.string('/tmp/malgeri/superPointing.root')
                               )

process.this_is_the_end = cms.EndPath(process.out)



