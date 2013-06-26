import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
"rfio:/castor/cern.ch/cms/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/065/1C27DDD3-84DD-DE11-AFA2-001617C3B66C.root",
"rfio:/castor/cern.ch/cms/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/065/20080E55-84DD-DE11-AB73-001D09F23F2A.root",
"rfio:/castor/cern.ch/cms/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/065/24952E6F-86DD-DE11-AC9D-0030486730C6.root",
"rfio:/castor/cern.ch/cms/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/065/26C49DD5-84DD-DE11-86B7-0016177CA778.root",
"rfio:/castor/cern.ch/cms/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/065/386CA562-86DD-DE11-A838-001D09F251BD.root",
"rfio:/castor/cern.ch/cms/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/065/38A1ADD6-84DD-DE11-B058-0019DB29C614.root",
"rfio:/castor/cern.ch/cms/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/065/58B694D7-85DD-DE11-BE70-001D09F29533.root",
"rfio:/castor/cern.ch/cms/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/065/7C925D72-86DD-DE11-8675-0019B9F72F97.root",
"rfio:/castor/cern.ch/cms/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/065/80AAEAD4-84DD-DE11-AD8C-001617C3B79A.root",
"rfio:/castor/cern.ch/cms/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/065/86E5AED4-84DD-DE11-BE26-001617DBCF6A.root",
"rfio:/castor/cern.ch/cms/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/065/8845DC7A-86DD-DE11-AFB3-001D09F24024.root",
"rfio:/castor/cern.ch/cms/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/065/AA0A257D-86DD-DE11-96E9-001D09F24259.root",
"rfio:/castor/cern.ch/cms/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/065/B0A1D8D3-84DD-DE11-B4FF-001617C3B6CE.root",
"rfio:/castor/cern.ch/cms/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/065/D08E327A-86DD-DE11-88EA-001D09F2447F.root",
"rfio:/castor/cern.ch/cms/store/express/BeamCommissioning09/ExpressPhysics/FEVT/v2/000/123/065/EE8DD2D7-84DD-DE11-907B-001617C3B65A.root"
),
   secondaryFileNames = cms.untracked.vstring(
)


#        '/store/data/Commissioning08/Cosmics/RAW/v1/000/069/578/085EFED4-E5AB-DD11-9ACA-001617C3B6FE.root')
)




process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/DPGAnalysis/Skims/python/RecoTrack_cfg.py,v $'),
    annotation = cms.untracked.string('At least two general track or one pixel track or one pixelLess track')
)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration/StandardSequences/GeometryIdeal_cff')


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR09_P_V6::All' 


process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")




process.generalTracksFilter = cms.EDFilter("TrackCountFilter",
                                                      src = cms.InputTag('generalTracks'),
                                                      minNumber = cms.uint32(2) 
                                                      )
process.pixelLessTracksFilter = cms.EDFilter("TrackCountFilter",
                                                      src = cms.InputTag('ctfPixelLess'),
                                                      minNumber = cms.uint32(1) 
                                                      )
process.pixelTracksFilter = cms.EDFilter("TrackCountFilter",
                                                      src = cms.InputTag('pixelTracks'),
                                                      minNumber = cms.uint32(1) 
                                                      )

process.generalTracksPath = cms.Path(process.generalTracksFilter)
process.pixelTracksPath = cms.Path(process.pixelLessTracksFilter)
process.pixelLessTracksPath = cms.Path(process.pixelTracksFilter)


process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('keep *','drop *_MEtoEDMConverter_*_*'),
                               SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('generalTracksPath',
    										'pixelTracksPath',
										'pixelLessTracksPath')),
                               dataset = cms.untracked.PSet(
			                 dataTier = cms.untracked.string('RAW-RECO'),
                                         filterName = cms.untracked.string('RecoTracks')),
                               fileName = cms.untracked.string('generalTracks.root')
                               )

process.this_is_the_end = cms.EndPath(process.out)
