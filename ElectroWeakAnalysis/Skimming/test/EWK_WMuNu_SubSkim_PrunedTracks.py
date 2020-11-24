import FWCore.ParameterSet.Config as cms

# Process name
process = cms.Process("WMuNuAODSkim")

# Source, events to process
process.source = cms.Source("PoolSource",
      fileNames = cms.untracked.vstring (
            #"file:/data4/InclusiveMu15_Summer09-MC_31X_V3_AODSIM-v1/0024/C2F408ED-E181-DE11-8949-0030483344E2.root"
            "file:/data4/Wmunu_Summer09-MC_31X_V3_AODSIM-v1/0009/F82D4260-507F-DE11-B5D6-00093D128828.root"
      )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
#process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

# Log information
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('ERROR')
    ),
    debugModules = cms.untracked.vstring('corMetWMuNus')
)

process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# Trigger filter (apply if for safety even if it may be redundant on a SD input)
import HLTrigger.HLTfilters.hltHighLevel_cfi
process.EWK_WMuNuHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
process.EWK_WMuNuHLTFilter.HLTPaths = ["HLT_Mu9", "HLT_DoubleMu3"]
#-> Use the following line for the 8E29 menu:
process.EWK_WMuNuHLTFilter.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT8E29")
#-> Use the following line for the 1E31 menu:
#process.EWK_WMuNuHLTFilter.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")

# Make sure that we have any muon there and optionally apply quality, eta, pt cuts
process.goodMuons = cms.EDFilter("CandViewSelector",
        src = cms.InputTag("muons")
      , filter = cms.bool(True)
      , cut = cms.string('pt>0')
      #, cut = cms.string('isGlobalMuon = 1 & abs(eta) < 2.5 & pt > 15')
)

# Tracks filtered
process.goodAODTracks = cms.EDFilter("TrackSelector",
    src = cms.InputTag("generalTracks"),
    #cut = cms.string('pt > 5.0 & numberOfValidHits>7')
    cut = cms.string('pt > 5.0')
)

# Electrons filtered
process.goodAODElectrons = cms.EDFilter("GsfElectronSelector",
    src = cms.InputTag("gsfElectrons"),
    cut = cms.string('pt > 5.0')
)

# Photons filtered
process.goodAODPhotons = cms.EDFilter("PhotonSelector",
    src = cms.InputTag("photons"),
    cut = cms.string('et > 5.0')
)

# For creaton of WMuNu Candidates
#process.load("ElectroWeakAnalysis.WMuNu.wmunusProducer_cfi")

# Main path
process.EWK_WMuNuSkimPath = cms.Path(
        process.EWK_WMuNuHLTFilter 
      + process.goodMuons
      + process.goodAODTracks
      + process.goodAODElectrons
      + process.goodAODPhotons
      #+ process.allWMuNus
)

# Choose collections for output
process.load("Configuration.EventContent.EventContent_cff")

# Write either a full AOD ...
#process.EWK_WMuNuEventContent = cms.PSet(outputCommands=process.AODEventContent.outputCommands)
# ... OR a reduced one with ~ 1/10 of original AOD size
process.EWK_WMuNuEventContent = cms.PSet(outputCommands=cms.untracked.vstring('drop *'))
process.EWK_WMuNuEventContent.outputCommands.extend(
      cms.untracked.vstring(
              'keep *_offlineBeamSpot_*_*'
            , 'keep *_TriggerResults_*_HLT8E29'
            , 'keep *_hltTriggerSummaryAOD_*_HLT8E29'
            , 'keep *_muons_*_*'
            , 'keep recoTracks_globalMuons_*_*'
            , 'keep recoTracks_standAloneMuons_*_*'
            , 'keep *_met_*_*'
            , 'keep *_corMetGlobalMuons_*_*'
            , 'keep *_tcMet_*_*'
            , 'keep *_pfMet_*_*'
            , 'keep *_antikt5CaloJets_*_*'
            , 'keep *_antikt5PFJets_*_*'
            ####
            , 'keep *_goodAODTracks_*_*'
            , 'keep *_goodAODElectrons_*_*'
            , 'keep *_goodAODPhotons_*_*'
            #, 'keep recoWMuNuCandidates_*_*_*'
      )
)

# Output
process.EWK_WMuNuSkimOutputModule = cms.OutputModule("PoolOutputModule"
      , process.EWK_WMuNuEventContent
      , dropMetaDataForDroppedData = cms.untracked.bool(True)
      , SelectEvents = cms.untracked.PSet(
            SelectEvents = cms.vstring('EWK_WMuNuSkimPath')
        )
      , dataset = cms.untracked.PSet(
              filterName = cms.untracked.string('EWKWMunuSkim')
            , dataTier = cms.untracked.string('USER')
        )
      , fileName = cms.untracked.string('EWK_WMuNu_SubSkim.root')
)

# End path
process.outpath = cms.EndPath(process.EWK_WMuNuSkimOutputModule)
