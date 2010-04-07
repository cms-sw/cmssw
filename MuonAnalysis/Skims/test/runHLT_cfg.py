# the file OnData_HLT_OFFLINE.py that is imported below has been obtained with the command:
# $CMSSW_RELEASE_BASE/src/HLTrigger/Configuration/test/getHLT.py --offline --data orcoff:/cdaq/physics/firstCollisions10/v3.0/HLT_7TeV/V1 OFFLINE 
from OnData_HLT_OFFLINE import *

# make a filter on the muonTracksSkim, l1MuBitsSkim bits available from the muon skim (SKIM)
process.filter =cms.EDFilter("HLTHighLevel",
     TriggerResultsTag = cms.InputTag("TriggerResults","","SKIM"),
     HLTPaths = cms.vstring('muonTracksSkim','l1MuBitsSkim'),           # provide list of HLT paths (or patterns) you want
     eventSetupPathsKey = cms.string(''), # not empty => use read paths from AlCaRecoTriggerBitsRcd via this key
     andOr = cms.bool(True),             # how to deal with multiple triggers: True (OR) accept if ANY is true, False (AND) accept if ALL are true
     throw = cms.bool(False)    # throw exception on unknown path names
)

# insert the filter in the beginning of each path in order to run HLT only on these events
process.filterSequence = cms.Sequence(process.filter)
for path in process.paths:
    getattr(process,path)._seq = process.filterSequence*getattr(process,path)._seq

# make a path to filter the output on
process.filterPath = cms.Path(process.filter)

process.load('Configuration.EventContent.EventContent_cff')
process.output = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    fileName = cms.untracked.string('file:hlt.root'),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('RAW-HLTDEBUG-RECO'),
        filterName = cms.untracked.string('')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring("filterPath")
    )
)

# remove the endpaths from the default configuration and add only ours
import FWCore.ParameterSet.DictTypes
process.__dict__['_Process__endpaths'] = FWCore.ParameterSet.DictTypes.SortedKeysDict()
process.end = cms.EndPath(process.output)

# remove the prescales
process.PrescaleService.prescaleTable = cms.VPSet()

process.source.fileNames = cms.untracked.vstring(
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/661/AA3902F7-BE41-DF11-A2A3-00E08178C179.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/661/9ECF25DB-C341-DF11-A011-00E0817918C1.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/661/1CAF665C-BD41-DF11-ACA4-001A64789E6C.root'
)

