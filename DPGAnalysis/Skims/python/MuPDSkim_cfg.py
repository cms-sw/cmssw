import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.2 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/DPGAnalysis/Skims/python/MuPDSkim_cfg.py,v $'),
    annotation = cms.untracked.string('Mu skim')
)

#
#
# This is for testing purposes.
#
#
##run143960
process.source = cms.Source("PoolSource",
                           fileNames = cms.untracked.vstring('/store/data/Run2010A/Mu/RECO/v4/000/144/114/9C954151-32B4-DF11-BB88-001D09F27003.root',
                                                             '/store/data/Run2010A/Mu/RECO/v4/000/144/114/5C9CA515-20B4-DF11-9D62-0030487A3DE0.root',
                                                             '/store/data/Run2010A/Mu/RECO/v4/000/144/114/00CA69A9-1EB4-DF11-B869-0030487CD6D2.root'),
                           secondaryFileNames = cms.untracked.vstring('/store/data/Run2010A/Mu/RAW/v1/000/144/114/70E434AF-04B4-DF11-B178-0030487CD718.root',
                                                                      '/store/data/Run2010A/Mu/RAW/v1/000/144/114/C25ACF12-F3B3-DF11-8BB6-0030487CD700.root',
                                                                      '/store/data/Run2010A/Mu/RAW/v1/000/144/114/643C84A4-F1B3-DF11-8809-003048F0258C.root')
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
process.GlobalTag.globaltag = 'GR10_P_V10::All'

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

process.load("DPGAnalysis.Skims.WZMuSkim_cff")

process.ZMuSkimPath      = cms.Path(process.diMuonSelSeq)
process.WtcMetSkimPath   = cms.Path(process.tcMetWMuNuSeq)
process.WpfMetSkimPath   = cms.Path(process.pfMetWMuNuSeq)


process.SKIMStreamWZMu = cms.OutputModule("PoolOutputModule",
                                          # splitLevel = cms.untracked.int32(0),
                                          outputCommands = process.FEVTEventContent.outputCommands,
                                          fileName = cms.untracked.string('WZMuStream.root'),
                                          dataset = cms.untracked.PSet(dataTier = cms.untracked.string('RAW-RECO'),
                                                                       filterName = cms.untracked.string('WZMuFilter')),
                                          SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('ZMuSkimPath',
                                                                                                       'WtcMetSkimPath',
                                                                                                       'WpfMetSkimPath')
                                                                            ))


#################################logerrorharvester############################################

process.load("FWCore.Modules.logErrorFilter_cfi")
from Configuration.StandardSequences.RawToDigi_Data_cff import gtEvmDigis

process.gtEvmDigis = gtEvmDigis.clone()
process.stableBeam = cms.EDFilter("HLTBeamModeFilter",
                                  L1GtEvmReadoutRecordTag = cms.InputTag("gtEvmDigis"),
                                  AllowedBeamMode = cms.vuint32(11)
                                  )

process.logerrorpath=cms.Path(process.gtEvmDigis+process.stableBeam+process.logErrorFilter)

process.outlogerr = cms.OutputModule("PoolOutputModule",
                                     outputCommands =  process.FEVTEventContent.outputCommands,
                                     fileName = cms.untracked.string('logerror_filter.root'),
                                     dataset = cms.untracked.PSet(dataTier = cms.untracked.string('RAW-RECO'),
                                                                  filterName = cms.untracked.string('Skim_logerror')),
                                     SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring("logerrorpath")
                                                                       ))

#======================

process.options = cms.untracked.PSet(
 wantSummary = cms.untracked.bool(True)
)

process.outpath = cms.EndPath(process.outlogerr+process.SKIMStreamWZMu)
