import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.5 $'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/DPGAnalysis/Skims/python/EGPDSkim_cfg.py,v $'),
    annotation = cms.untracked.string('EGamma skim')
)

#
#
# This is for testing purposes.
#
#
##run143960
process.source = cms.Source("PoolSource",
                           fileNames = cms.untracked.vstring(
    '/store/data/Run2010A/EG/RECO/v4/000/143/960/84DEE17A-44B1-DF11-B844-001D09F29849.root'
   ),
                           secondaryFileNames = cms.untracked.vstring(
'/store/data/Run2010A/EG/RAW/v1/000/143/960/C40C9318-0FB1-DF11-A974-0030487CBD0A.root')
   )

process.source.inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
    )


#------------------------------------------
# Load standard sequences.
#------------------------------------------
process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration/StandardSequences/GeometryIdeal_cff')


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR10_P_V8::All'

process.load("Configuration/StandardSequences/RawToDigi_Data_cff")
process.load("Configuration/StandardSequences/Reconstruction_cff")
process.load('Configuration/EventContent/EventContent_cff')

#drop collections created on the fly
process.FEVTEventContent.outputCommands.append("drop *_MEtoEDMConverter_*_*")
process.FEVTEventContent.outputCommands.append("drop *_*_*_SKIM")
#
#  Load common sequences
#
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskAlgoTrigConfig_cff')
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')

#################################WZFilter############################################

process.hltFilter = cms.EDFilter("HLTHighLevel",
                                 TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
                                 HLTPaths = cms.vstring(
    #    "HLT_Photon15_L1R",
    #    "HLT_Photon15_Cleaned_L1R",
    #    "HLT_Photon20_Cleaned_L1R",
    "HLT_Ele15_LW_L1R",
    "HLT_Ele15_SW_L1R",
    "HLT_Ele15_SW_CaloEleId_L1R",
    "HLT_Ele17_SW_CaloEleId_L1R",
    "HLT_Ele17_SW_L1R",
    "HLT_Ele17_SW_TightEleId_L1R",
    "HLT_Ele17_SW_TightCaloEleId_SC8HE_L1R"
    ),
                                 eventSetupPathsKey = cms.string(''),
                                 andOr = cms.bool(True),
                                 throw = cms.bool(False),
                                 saveTags = cms.bool(False)
                                 )

process.load("DPGAnalysis/Skims/WZinterestingEventFilter_cfi")

process.WZfilter = cms.Path(process.hltFilter*process.WZInterestingEventSelector)

# Output definition
process.outWZfilter = cms.OutputModule("PoolOutputModule",
                                       # splitLevel = cms.untracked.int32(0),
                                       outputCommands = process.FEVTEventContent.outputCommands,
                                       fileName = cms.untracked.string('/tmp/azzi/EGMWZ_filter.root'),
                                       dataset = cms.untracked.PSet(dataTier = cms.untracked.string('RAW-RECO'),
                                                                    filterName = cms.untracked.string('EGMWZFilter')),
                                       SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('WZfilter')
                                                                         ))


#################################logerrorharvester############################################

process.load("FWCore.Modules.logErrorFilter_cfi")
from Configuration.StandardSequences.RawToDigi_Data_cff import gtEvmDigis

process.gtEvmDigis = gtEvmDigis.clone()
process.stableBeam = cms.EDFilter("HLTBeamModeFilter",
                                  L1GtEvmReadoutRecordTag = cms.InputTag("gtEvmDigis"),
                                  AllowedBeamMode = cms.vuint32(11),
                                  saveTags = cms.bool(False)
                                  )

process.logerrorpath=cms.Path(process.gtEvmDigis+process.stableBeam+process.logErrorFilter)

process.outlogerr = cms.OutputModule("PoolOutputModule",
                                     outputCommands =  process.FEVTEventContent.outputCommands,
                                     fileName = cms.untracked.string('/tmp/azzi/logerror_filter.root'),
                                     dataset = cms.untracked.PSet(dataTier = cms.untracked.string('RAW-RECO'),
                                                                  filterName = cms.untracked.string('Skim_logerror')),
                                     SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring("logerrorpath")
                                                                       ))

#======================

process.options = cms.untracked.PSet(
 wantSummary = cms.untracked.bool(True)
)

process.outpath = cms.EndPath(process.outlogerr+process.outWZfilter)
