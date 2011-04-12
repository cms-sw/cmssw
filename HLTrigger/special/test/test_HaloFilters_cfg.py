#
# This python script is the basis for MIB HLT path testing
# 
# Only the developed path are runned on the RAW data sample
#
# We are using GRun_data version of the HLT menu
#
# SV (viret@in2p3.fr): 18/01/2011
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
useGlobalTag = 'GR_R_311_V1::All'
#useGlobalTag = 'START311_V1::All'

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.options = cms.untracked.PSet(
    SkipEvent = cms.untracked.vstring('ProductNotFound')
)

# Input source (a raw data file from the Commissioning dataset)

process.source = cms.Source("PoolSource",
                            noEventSort = cms.untracked.bool(True),
                            duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/data/Run2010B/Commissioning/RAW/v1/000/147/043/0C1114D5-E5CD-DF11-8FF5-001D09F2546F.root')
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

process.HLTBeginSequence = cms.Sequence( process.hltTriggerType + process.HLTL1UnpackerSequence )

process.HLTDoLocalPixel = cms.Sequence( process.hltSiPixelDigis + process.hltSiPixelClusters)





process.HLTDoLocalStrips= cms.Sequence( process.hltSiStripRawToClustersFacility + process.hltSiStripClusters)

#process.HLTDoLocalStrips= cms.Sequence( process.hltSiStripRawToClustersFacility)



process.hltPixelActivityFilter = cms.EDFilter( "HLTPixelActivityFilter",
   inputTag    = cms.InputTag( "hltSiPixelClusters" ),
   saveTag     = cms.untracked.bool( False ),
   minClusters = cms.uint32( 0 ),
   maxClusters = cms.uint32( 10 )                                    
)

process.hltTrackerHaloFilter = cms.EDFilter( "HLTTrackerHaloFilter",
   inputTag           = cms.InputTag( "hltSiStripClusters" ),
   saveTag            = cms.untracked.bool( False ),
   MaxClustersTECp    = cms.int32(50),
   MaxClustersTECm    = cms.int32(50),
   SignalAccumulation = cms.int32(5),
   MaxClustersTEC     = cms.int32(60),
   MaxAccus           = cms.int32(4),
   FastProcessing     = cms.int32(1)
)


# The test path


process.HLT_BeamHalo = cms.Path( process.HLTBeginSequence  + process.hltL1sL1BptxXORBscMinBiasOR  + process.HLTDoLocalPixel + process.hltPixelActivityFilter + process.HLTDoLocalStrips + process.hltTrackerHaloFilter + process.HLTEndSequence )


process.m_HLTSchedule = cms.Schedule( *(process.HLTriggerFirstPath, process.HLT_BeamHalo, process.HLTriggerFinalPath, process.HLTAnalyzerEndpath ))


#Deal with the global tag
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.connect   = 'frontier://FrontierProd/CMS_COND_31X_GLOBALTAG'
process.GlobalTag.pfnPrefix = cms.untracked.string('frontier://FrontierProd/')
process.GlobalTag.globaltag = useGlobalTag

# Path and EndPath definitions
process.endjob_step  = cms.Path(process.endOfProcess)
process.out_step     = cms.EndPath( process.hltTimer + process.output)


# Schedule definition

process.schedule = cms.Schedule(process.m_HLTSchedule)

process.schedule.extend([process.endjob_step,process.out_step])

