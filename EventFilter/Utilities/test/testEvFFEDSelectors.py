import FWCore.ParameterSet.Config as cms

process = cms.Process('TEST')

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing('analysis')
options.setDefault('inputFiles', [
  'root://eoscms.cern.ch//eos/cms/store/group/dpg_trigger/comm_trigger/TriggerStudiesGroup/STORM/RAW/Run2022B_HLTPhysics0_run355558/cd851cf4-0fca-4d76-b80e-1d33e1371929.root',
])
options.setDefault('maxEvents', 10)
options.parseArguments()

# set max number of input events
process.maxEvents.input = options.maxEvents

# initialize MessageLogger and output report
process.options.wantSummary = False
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 100 # only report every 100th event start
process.MessageLogger.cerr.enableStatistics = False # enable "MessageLogger Summary" message
process.MessageLogger.cerr.threshold = 'INFO' # change to 'WARNING' not to show INFO-level messages
## enable reporting of INFO-level messages (default is limit=0, i.e. no messages reported)
#process.MessageLogger.cerr.INFO = cms.untracked.PSet(
#    reportEvery = cms.untracked.int32(1), # every event!
#    limit = cms.untracked.int32(-1)       # no limit!
#)

###
### Source (input file)
###
process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring(options.inputFiles)
)
print('process.source.fileNames =', process.source.fileNames)

###
### Path (FEDRAWData producers)
###
_siStripFEDs = [foo for foo in range(50, 489+1)]

from EventFilter.Utilities.EvFFEDSelector_cfi import EvFFEDSelector as _EvFFEDSelector
process.rawDataSiStripV1 = _EvFFEDSelector.clone(
  inputTag = 'rawDataCollector',
  fedList = _siStripFEDs,
)

from EventFilter.Utilities.EvFFEDExcluder_cfi import EvFFEDExcluder as _EvFFEDExcluder
process.rawDataSiStripV2 = _EvFFEDExcluder.clone(
  src = 'rawDataCollector',
  fedsToExclude = [foo for foo in range(4096+1) if foo not in _siStripFEDs],
)

process.rawDataSelectionPath = cms.Path(
    process.rawDataSiStripV1
  + process.rawDataSiStripV2
)

###
### EndPath (output file)
###
process.rawDataOutputModule = cms.OutputModule('PoolOutputModule',
    fileName = cms.untracked.string('file:tmp.root'),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep FEDRawDataCollection_*_*_*',
        'keep edmTriggerResults_*_*_*',
        'keep triggerTriggerEvent_*_*_*',
    )
)

process.outputEndPath = cms.EndPath( process.rawDataOutputModule )
