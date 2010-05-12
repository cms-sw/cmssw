# save to file the HLT menu from the database with a given HLT key  
import os
os.system('$CMSSW_RELEASE_BASE/src/HLTrigger/Configuration/test/getHLT.py --force --offline --data orcoff:/cdaq/physics/firstCollisions10/v5.1/HLT_7TeV_HR/V1 MUON')

# import the menu
from OnData_HLT_MUON import *

# remove the prescales
process.PrescaleService.prescaleTable = cms.VPSet()

# test input
process.source.fileNames = cms.untracked.vstring(
       '/store/data/Commissioning10/MinimumBias/RAW-RECO/v9/000/133/928/82A98A96-7B51-DF11-8BB9-002481E14E82.root',
       '/store/data/Commissioning10/MinimumBias/RAW-RECO/v9/000/133/928/7AAB9CC2-7551-DF11-8A68-003048D476FA.root',
       '/store/data/Commissioning10/MinimumBias/RAW-RECO/v9/000/133/928/7A070DDA-7351-DF11-B779-001A64789E48.root',
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

# Open mass window and lower P cut in the Jpsi triggers for studies
process.hltMuTrackJpsiPixelTrackSelector.MinTrackP = cms.double( 2.2 )
process.hltMu0TrackJpsiPixelMassFiltered.MinTrackP = cms.double( 2.2 )
process.hltMu0TrackJpsiTrackMassFiltered.MinTrackP = cms.double( 2.5 )
process.hltMu0TrackJpsiTrackMassFiltered.MinMasses = cms.vdouble( 2.6 )
process.hltMu0TrackJpsiTrackMassFiltered.MaxMasses = cms.vdouble( 3.6 )
process.hltMu3TrackJpsiPixelMassFiltered.MinTrackP = cms.double( 2.2 )
process.hltMu3TrackJpsiTrackMassFiltered.MinTrackP = cms.double( 2.5 )
process.hltMu3TrackJpsiTrackMassFiltered.MinMasses = cms.vdouble( 2.6 )
process.hltMu3TrackJpsiTrackMassFiltered.MaxMasses = cms.vdouble( 3.6 )
process.hltMu5TrackJpsiPixelMassFiltered.MinTrackP = cms.double( 2.2 )
process.hltMu5TrackJpsiTrackMassFiltered.MinTrackP = cms.double( 2.5 )
process.hltMu5TrackJpsiTrackMassFiltered.MinMasses = cms.vdouble( 2.6 )
process.hltMu5TrackJpsiTrackMassFiltered.MaxMasses = cms.vdouble( 3.6 )

