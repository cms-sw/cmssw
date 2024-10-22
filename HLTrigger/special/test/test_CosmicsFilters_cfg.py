#
# This python script is the basis for Cosmics HLT path testing
# 
# Only the developed path are runned on the RAW data sample
#
# We are using GRun_data version of the HLT menu
#
# SV (viret@in2p3.fr): 04/02/2011
#

import FWCore.ParameterSet.Config as cms

process = cms.Process('HLT2')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryIdeal_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('HLTrigger.Configuration.HLT_GRun_data_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.EventContent.EventContent_cff')


# To be adapted to the release
#useGlobalTag = 'GR_R_311_V1::All'
#useGlobalTag = 'START311_V2::All'
useGlobalTag = 'GR_P_V14::All' 

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.options = cms.untracked.PSet(
    SkipEvent = cms.untracked.vstring('ProductNotFound')
)

# Input source (a raw data file from the Commissioning dataset)
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.source = cms.Source("PoolSource",
                            noEventSort = cms.untracked.bool(True),
                            duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/data/Run2011A/Cosmics/RAW/v1/000/161/439/6A9B822C-3958-E011-9BFF-00304879FA4C.root')
                            #fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/user/t/tschudi/February11/RelValCosmics_withPU/RelValCosmic_withPU.root ')
                            
)


# Output module (keep only the stuff necessary to the timing module)

process.output = cms.OutputModule("PoolOutputModule",
                                  splitLevel = cms.untracked.int32(0),
                                  outputCommands = cms.untracked.vstring( 'drop *', 'keep HLTPerformanceInfo_*_*_*'),
                                  fileName = cms.untracked.string('HLT.root'),
                                  dataset = cms.untracked.PSet(
    dataTier = cms.untracked.string('RECO'),
    filterName = cms.untracked.string('')
    )
)


# Timer

process.PathTimerService  = cms.Service( "PathTimerService" )
process.hltTimer          = cms.EDProducer( "PathTimerInserter" )


# Then we define the info necessary to the paths



process.cosmicsNavigationSchoolESProducer = cms.ESProducer( "NavigationSchoolESProducer",
    ComponentName = cms.string( "CosmicNavigationSchool" ),
    appendToDataLabel = cms.string( "" )
)


process.hltTrackSeedMultiplicityFilter = cms.EDFilter( "HLTTrackSeedMultiplicityFilter",
   inputTag    = cms.InputTag( "hltRegionalCosmicTrackerSeeds" ),
   saveTags = cms.bool( False ),
   minSeeds = cms.uint32( 1 ),
   maxSeeds = cms.uint32( 100000 )                                    
)

# Seeding process

process.hltRegionalCosmicTrackerSeeds = cms.EDProducer( "SeedGeneratorFromRegionHitsEDProducer",
    ClusterCheckPSet = cms.PSet( 
        MaxNumberOfStripClusters = cms.uint32( 50000 ),
        ClusterCollectionLabel = cms.InputTag( "hltSiStripClusters" ),
        doClusterCheck = cms.bool( False ),
        PixelClusterCollectionLabel = cms.InputTag( "hltSiPixelClusters" ),
        MaxNumberOfPixelClusters = cms.uint32( 10000 )
        ),
    RegionFactoryPSet = cms.PSet(                                 
        ComponentName = cms.string( "CosmicRegionalSeedGenerator" ),
        RegionPSet = cms.PSet(
            ptMin          = cms.double( 5.0 ),
            rVertex        = cms.double( 5 ),
            zVertex        = cms.double( 5 ),
            deltaEtaRegion = cms.double( 0.3 ),
            deltaPhiRegion = cms.double( 0.3 ),
            precise        = cms.bool( True ),
            measurementTrackerName = cms.string('hltESPMeasurementTracker')
            ),
        ToolsPSet = cms.PSet(
            thePropagatorName = cms.string("hltESPAnalyticalPropagator"),
            regionBase        = cms.string("seedOnL2Muon")
            ),
        CollectionsPSet = cms.PSet(
            recoMuonsCollection      = cms.InputTag("muons"), 
            recoTrackMuonsCollection = cms.InputTag("cosmicMuons"), 
            recoL2MuonsCollection    = cms.InputTag("hltL2MuonCandidatesNoVtx")
            ),
        RegionInJetsCheckPSet = cms.PSet(
            doJetsExclusionCheck   = cms.bool( False ),
            deltaRExclusionSize    = cms.double( 0.3 ),
            jetsPtMin              = cms.double( 5 ),
            recoCaloJetsCollection = cms.InputTag("hltIterativeCone5CaloJets")
            )
        ),
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string( "GenericPairGenerator" ),
        LayerPSet = cms.PSet(
            TOB = cms.PSet(        
                TTRHBuilder = cms.string('hltESPTTRHBWithTrackAngle')
                ),
            layerList = cms.vstring('TOB6+TOB5',
                                    'TOB6+TOB4', 
                                    'TOB6+TOB3',
                                    'TOB5+TOB4',
                                    'TOB5+TOB3',
                                    'TOB4+TOB3')
            )
        ),             
    SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) ),
    SeedCreatorPSet = cms.PSet( 
        ComponentName = cms.string( "CosmicSeedCreator" ),
        propagator    = cms.string( "PropagatorWithMaterial" ),
        maxseeds      = cms.int32( 100000 )
        ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" )
)


# Define the sequences to be included in the path

process.HLTRegionalCosmicSeeding = cms.Sequence( process.hltSiPixelDigis + process.hltSiPixelClusters + process.hltSiStripRawToClustersFacility + process.hltSiStripClusters + process.hltRegionalCosmicTrackerSeeds + process.hltTrackSeedMultiplicityFilter )


# And finally the path to test

#process.HLT_TrackerCosmics_RegionalCosmicTracking = cms.Path( process.HLTBeginSequence + process.hltL1sTrackerCosmics + process.hltPreTrackerCosmics + process.hltTrackerCosmicsPattern + process.hltL1sL1SingleMu0 + process.hltPreMu0 + process.hltSingleMu0L1Filtered + process.HLTL2muonrecoSequenceNoVtx + process.HLTRegionalCosmicSeeding + process.HLTEndSequence )

process.HLT_TrackerCosmics_RegionalCosmicTracking = cms.Path( process.HLTBeginSequence + process.hltL1sTrackerCosmics + process.hltPreTrackerCosmics + process.hltTrackerCosmicsPattern + process.HLTL2muonrecoSequenceNoVtx + process.HLTRegionalCosmicSeeding + process.HLTEndSequence )


#Deal with the global tag
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.connect   = 'frontier://FrontierProd/CMS_COND_31X_GLOBALTAG'
process.GlobalTag.pfnPrefix = cms.untracked.string('frontier://FrontierProd/')
process.GlobalTag.globaltag = useGlobalTag

# Path and EndPath definitions
process.endjob_step  = cms.Path(process.endOfProcess)
process.out_step     = cms.EndPath( process.hltTimer + process.output)


# Schedule definition
process.schedule = cms.Schedule(*( process.HLTriggerFirstPath, process.HLT_TrackerCosmics_RegionalCosmicTracking, process.HLTriggerFinalPath, process.HLTAnalyzerEndpath ))
process.schedule.extend([process.endjob_step,process.out_step])
