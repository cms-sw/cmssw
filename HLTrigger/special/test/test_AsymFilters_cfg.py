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
useGlobalTag = 'GR_R_310_V3::All'

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
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
process.HLTDoLocalHF    = cms.Sequence( process.hltHcalDigis + process.hltHfreco )

process.hltPixelAsymmetryFilter = cms.EDFilter( "HLTPixelAsymmetryFilter",
                                               inputTag  = cms.InputTag( "hltSiPixelClusters" ),
                                               MinAsym   = cms.double( 0. ),     # minimum asymmetry 
                                               MaxAsym   = cms.double( 1. ),     # maximum asymmetry
                                               MinCharge = cms.double( 4000. ),  # minimum charge for a cluster to be selected (in e-)
                                               MinBarrel = cms.double( 10000. ), # minimum average charge in the barrel (bpix, in e-)
                                               ) 

process.hltHFAsymmetryFilter = cms.EDFilter( "HLTHFAsymmetryFilter",
                                            ECut_HF         = cms.double( 3.0 ),  # minimum energy for a cluster to be selected
                                            OS_Asym_max     = cms.double( 0.2 ),  # Opposite side asymmetry maximum value
                                            SS_Asym_min     = cms.double( 0.8 ),  # Same side asymmetry minimum value
                                            HFHitCollection = cms.InputTag( "hltHfreco" )
                                            ) 


# The test paths


process.HLT_L1_BSC_BeamGas = cms.Path( process.HLTBeginSequence  + process.hltL1sL1BptxXORBscMinBiasOR  + process.HLTDoLocalPixel + process.hltPixelAsymmetryFilter + process.HLTEndSequence )

process.HLT_L1_HF_BeamGas = cms.Path( process.HLTBeginSequence   + process.hltL1sL1BptxXORBscMinBiasOR  + process.HLTDoLocalHF + process.hltHFAsymmetryFilter + process.HLTEndSequence )


#Deal with the global tag
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.connect   = 'frontier://FrontierProd/CMS_COND_31X_GLOBALTAG'
process.GlobalTag.pfnPrefix = cms.untracked.string('frontier://FrontierProd/')
process.GlobalTag.globaltag = useGlobalTag

# Path and EndPath definitions
process.endjob_step  = cms.Path(process.endOfProcess)
process.out_step     = cms.EndPath( process.hltTimer + process.output)


# Schedule definition
process.schedule = cms.Schedule(*( process.HLTriggerFirstPath, process.HLT_L1_BSC_BeamGas, process.HLT_L1_HF_BeamGas, process.HLTriggerFinalPath, process.HLTAnalyzerEndpath ))
process.schedule.extend([process.endjob_step,process.out_step])
