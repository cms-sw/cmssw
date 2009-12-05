import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.2 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/DPGAnalysis/Skims/python/CosmicsPDSkim_cfg.py,v $'),
    annotation = cms.untracked.string('Combined Cosmics skim')
)

#
#
# This is for testing purposes.
#
# run 122314 lumi 27
#
process.source = cms.Source("PoolSource",
fileNames = cms.untracked.vstring('/store/data/BeamCommissioning09/Cosmics/RECO/v2/000/122/314/6AECB986-87D8-DE11-A470-001D09F29169.root'),
secondaryFileNames = cms.untracked.vstring('/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/122/314/0C7140FD-7CD8-DE11-A515-001D09F2960F.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)



#------------------------------------------
# Load standard sequences.
#------------------------------------------
process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration/StandardSequences/GeometryIdeal_cff')


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR09_P_V7::All' 

process.load("Configuration/StandardSequences/RawToDigi_Data_cff")
process.load("Configuration/StandardSequences/ReconstructionCosmics_cff")
process.load('Configuration/EventContent/EventContentCosmics_cff')


######################################Super Pointing#################################################

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



process.outSP = cms.OutputModule("PoolOutputModule",
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
                               fileName = cms.untracked.string('superPointing.root')
                               )

###########################################################################################
#------------------------------------------
# parameters for the CSCSkim module
#------------------------------------------
process.load("DPGAnalysis/Skims/CSCSkim_cfi")


#set to minimum activity
process.cscSkim.minimumSegments = 1
process.cscSkim.minimumHitChambers = 1

# this is for filtering on HLT path
process.hltBeamHalo = cms.EDFilter("HLTHighLevel",
     TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
     HLTPaths = cms.vstring('HLT_CSCBeamHalo','HLT_CSCBeamHaloOverlapRing1','HLT_CSCBeamHaloOverlapRing','HLT_CSCBeamHaloRing2or3'), # provide list of HLT paths (or patterns) you want
     eventSetupPathsKey = cms.string(''), # not empty => use read paths from AlCaRecoTriggerBitsRcd via this key
     andOr = cms.bool(True),             # how to deal with multiple triggers: True (OR) accept if ANY is true, False (AND) accept if ALL are true
     throw = cms.bool(False)    # throw exception on unknown path names
 )

#### the path
process.cscHaloSkim = cms.Path(process.hltBeamHalo+process.cscSkim)

process.options = cms.untracked.PSet(
 wantSummary = cms.untracked.bool(True)
)


#### output 
process.outputBeamHaloSkim = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *','drop *_MEtoEDMConverter_*_*'),
    fileName = cms.untracked.string("cscskimEvents.root"),
    dataset = cms.untracked.PSet(
      dataTier = cms.untracked.string('RAW-RECO'),
      filterName = cms.untracked.string('CSCSkim_BeamHalo')
    ),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('cscHaloSkim'))
)

process.outpath = cms.EndPath(process.outSP+process.outputBeamHaloSkim)




 
