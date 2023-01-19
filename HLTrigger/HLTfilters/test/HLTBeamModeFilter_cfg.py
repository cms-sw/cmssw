#
# cfg file to test HLTBeamModeFilter
# it requires as input:
#   a RAW data file (with FED 812 included) or 
#   a digi data file, with L1GlobalTriggerEvmReadoutRecord product valid
#
# V.M. Ghete 2010-05-31
#
import FWCore.ParameterSet.Config as cms
import sys

useMC = False

process = cms.Process('TestHLTBeamModeFilter')

process.source = cms.Source('PoolSource', fileNames = cms.untracked.vstring())

if useMC:
    gtName = 'auto:run2_mc_l1stage1'
    process.source.fileNames = ['/store/relval/CMSSW_5_3_6-START53_V14/RelValProdMinBias/GEN-SIM-RAW/v2/00000/52000D8A-032A-E211-BC94-00304867BFA8.root']
else:
    gtName = 'auto:run2_data'
    process.source.fileNames = ['/store/data/Run2012A/MuEG/RAW/v1/000/191/718/14932935-E289-E111-830C-5404A6388697.root']

# load and configure modules via GlobalTag
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, gtName, '')

# number of events to be processed and source file
process.maxEvents.input = 100

from EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmUnpack_cfi import l1GtEvmUnpack as _l1GtEvmUnpack
process.gtEvmDigis = _l1GtEvmUnpack.clone(
    EvmGtInputTag = 'rawDataCollector',
    UnpackBxInEvent = 1,
    Verbosity = cms.untracked.int32(1) # set EVM unpacker to verbose
)

process.load('HLTrigger.HLTfilters.hltBeamModeFilter_cfi')
# replacing arguments for hltBeamModeFilter
#  InputTag for the L1 Global Trigger EVM readout record
#   gtDigis        GT Emulator
#   l1GtEvmUnpack  GT EVM Unpacker (default module name)
#   gtEvmDigis     GT EVM Unpacker in RawToDigi standard sequence
#
#   cloned GT unpacker in HLT = gtEvmDigis
process.hltBeamModeFilter.L1GtEvmReadoutRecordTag = 'gtEvmDigis'
# vector of allowed beam modes (see enumeration in header file for implemented values)
# default value: 11 (STABLE)
#process.hltBeamModeFilter.AllowedBeamMode = [11]
process.hltBeamModeFilter.AllowedBeamMode = [9, 10, 11]

process.p = cms.Path(process.gtEvmDigis * process.hltBeamModeFilter)

# services

# Message Logger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['gtEvmDigis', 'hltBeamModeFilter']
process.MessageLogger.L1GlobalTriggerEvmRawToDigi = dict()
process.MessageLogger.HLTBeamModeFilter = dict()

process.MessageLogger.cerr.threshold = 'DEBUG'
#process.MessageLogger.cerr.threshold = 'INFO'
#process.MessageLogger.cerr.threshold = 'WARNING'
#process.MessageLogger.cerr.threshold = 'ERROR'

process.MessageLogger.cerr.DEBUG = cms.untracked.PSet( limit = cms.untracked.int32(0) )
process.MessageLogger.cerr.INFO = cms.untracked.PSet( limit = cms.untracked.int32(0) )
process.MessageLogger.cerr.WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) )
process.MessageLogger.cerr.ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) )

process.MessageLogger.cerr.L1GlobalTriggerEvmRawToDigi = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
process.MessageLogger.cerr.HLTBeamModeFilter = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
