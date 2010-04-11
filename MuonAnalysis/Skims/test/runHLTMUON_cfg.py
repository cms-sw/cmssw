# save to file the HLT menu from the database with a given HLT key  
import os
os.system('$CMSSW_RELEASE_BASE/src/HLTrigger/Configuration/test/getHLT.py --force --offline --data orcoff:/cdaq/physics/firstCollisions10/v3.0/HLT_7TeV/V1 MUON')

# import the menu
from OnData_HLT_MUON import *

# remove the prescales
process.PrescaleService.prescaleTable = cms.VPSet()

# test input
process.source.fileNames = cms.untracked.vstring(
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/661/AA3902F7-BE41-DF11-A2A3-00E08178C179.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/661/9ECF25DB-C341-DF11-A011-00E0817918C1.root',
        '/store/data/Commissioning10/MinimumBias/RAW-RECO/v8/000/132/661/1CAF665C-BD41-DF11-ACA4-001A64789E6C.root'
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )    

# make a filter on the muonTracksSkim, l1MuBitsSkim bits available from the muon skim (SKIM)
process.filter =cms.EDFilter("HLTHighLevel",
     TriggerResultsTag = cms.InputTag("TriggerResults","","SKIM"),
     HLTPaths = cms.vstring('muonTracksSkim','l1MuBitsSkim'),           # provide list of HLT paths (or patterns) you want
     eventSetupPathsKey = cms.string(''), # not empty => use read paths from AlCaRecoTriggerBitsRcd via this key
     andOr = cms.bool(True),             # how to deal with multiple triggers: True (OR) accept if ANY is true, False (AND) accept if ALL are true
     throw = cms.bool(False)    # throw exception on unknown path names
)

# make a path to filter the output on
process.filterPath = cms.Path(process.filter)

# save this output
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

############## before 3.6.0 and 3.5.7 ##############
#add some missing output commands
process.output.outputCommands.extend([
    'keep *_hltMuTrackJpsiCtfTracks_*_*',
    'keep *_hltMuTrackJpsiCtfTrackCands_*_*',
    'keep *_hltMuTrackJpsiTrackSeeds_*_*',
    'keep *_hltMuTrackJpsiPixelTrackCands_*_*',
    'keep *_hltMuTrackJpsiPixelTrackSelector_*_*'
])
#other fixes
process.hltL2Mu0L2Filtered0.SaveTag = cms.untracked.bool( True )
process.hltSingleMu3L2Filtered3.SaveTag = cms.untracked.bool( True )
process.hltDiMuonL2PreFiltered0.SaveTag = cms.untracked.bool( True )
# define HLT_L2Mu5
process.hltL2Mu5L2Filtered5 = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltL1SingleMu0L1Filtered0" ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 5.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
process.HLT_L2Mu5 = cms.Path( process.HLTBeginSequenceBPTX + process.hltL1sL1SingleMuOpenL1SingleMu0L1SingleMu3 + process.hltL1SingleMu0L1Filtered0 + process.HLTL2muonrecoSequence + process.hltL2Mu5L2Filtered5 + process.HLTEndSequence )
########################################################

# remove the endpaths from the default configuration
import FWCore.ParameterSet.DictTypes
process.__dict__['_Process__endpaths'] = FWCore.ParameterSet.DictTypes.SortedKeysDict()

# add our endpath
process.end = cms.EndPath(process.output)

# add this path to run muon HLT reconstruction independently from the filters
process.muonRECO = cms.Path( process.HLTBeginSequence * process.HLTL2muonrecoSequence * process.HLTL2muonisorecoSequence * process.HLTL3muonrecoSequence * process.HLTL3muonisorecoSequence * process.HLTMuTrackJpsiPixelRecoSequence * process.HLTMuTrackJpsiTrackRecoSequence )

# run only these paths
process.schedule = cms.Schedule(
# process.HLTriggerFirstPath,
 process.HLT_L1MuOpen,
 process.HLT_L1MuOpen_NoBPTX,
 process.HLT_L1MuOpen_AntiBPTX,
 process.HLT_L1Mu,
 process.HLT_L1Mu20,
 process.HLT_L1DoubleMuOpen,
 process.HLT_L2Mu0,
 process.HLT_L2Mu3,
 process.HLT_L2Mu5,
 process.HLT_L2Mu9,
 process.HLT_L2Mu11,
 process.HLT_L2DoubleMu0,
 process.HLT_IsoMu3,
 process.HLT_Mu3,
 process.HLT_Mu5,
 process.HLT_Mu9,
 process.HLT_DoubleMu0,
 process.HLT_DoubleMu3,
 process.HLT_Mu0_L1MuOpen,
 process.HLT_Mu3_L1MuOpen,
 process.HLT_Mu5_L1MuOpen,
 process.HLT_Mu0_L2Mu0,
 process.HLT_Mu3_L2Mu0,
 process.HLT_Mu5_L2Mu0,
 process.HLT_Mu0_Track0_Jpsi,
 process.HLT_Mu3_Track0_Jpsi,
 process.HLT_Mu5_Track0_Jpsi,
 process.HLT_L2Mu0_NoVertex,
 process.HLT_TkMu3_NoVertex,
 process.muonRECO,
 process.filterPath,
 process.HLT_LogMonitor,
 process.HLTriggerFinalPath,
 process.end
)

