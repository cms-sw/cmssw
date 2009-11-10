import FWCore.ParameterSet.Config as cms

process = cms.Process("EwkZMuMuGolden")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.MessageLogger.cerr.threshold = ''
process.MessageLogger.cerr.FwkReport.reportEvery = 1

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('MC_31X_V3::All')
process.load("Configuration.StandardSequences.MagneticField_cff")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)



process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
    'rfio:/castor/cern.ch/cms/store/relval/CMSSW_3_4_0_pre1/RelValZMM/GEN-SIM-RECO/STARTUP31X_V8-v1/0007/CAE2081C-48B5-DE11-9161-001D09F29321.root',
    )
)

import copy

#####################################################
#                MUONS for ZMuMu                    #    
#####################################################


process.load("RecoMuon.MuonIsolationProducers.muIsolation_cff")




process.goodAODMuons = cms.EDFilter("CandViewRefSelector",
  src = cms.InputTag("muons"),
  cut = cms.string('pt > 20'),
  filter = cms.bool(True)                                
)

process.goodAODGlobalMuons = cms.EDFilter("CandViewRefSelector",
  src = cms.InputTag("goodAODMuons"),
  cut = cms.string('isGlobalMuon=1'),
  filter = cms.bool(True)                                
)

process.AODMuonIsoDepositCtfTk = cms.EDProducer("MuIsoDepositProducer",
    src                  = cms.InputTag("goodAODGlobalMuons"),
    IOPSet = cms.PSet (process.MIsoDepositViewIOBlock),
    ExtractorPSet        = cms.PSet( process.MIsoTrackExtractorCtfBlock )
)

#######  combiner modules ##########

process.dimuonsAOD = cms.EDFilter("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(True),
    cut = cms.string('mass > 0'),
    decay = cms.string("goodAODMuons@+ goodAODMuons@-")
)

process.dimuonsGlobalAOD = cms.EDFilter("CandViewRefSelector",
    src = cms.InputTag("dimuonsAOD"),
    cut = cms.string('charge = 0 & daughter(0).isGlobalMuon = 1 & daughter(1).isGlobalMuon = 1')
)


# dimuon filter
process.dimuonsFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("dimuonsAOD"),
    minNumber = cms.uint32(1)
)


import HLTrigger.HLTfilters.hltHighLevel_cfi

process.dimuonsHLTFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
# Add this to access 8E29 menu
#dimuonsHLTFilter.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT8E29")
# for 8E29 menu
#dimuonsHLTFilter.HLTPaths = ["HLT_Mu3", "HLT_DoubleMu3"]
# for 1E31 menu
#dimuonsHLTFilter.HLTPaths = ["HLT_Mu9", "HLT_DoubleMu3"]
process.dimuonsHLTFilter.HLTPaths = ["HLT_Mu9"]




########################################
#####          selection         #######
########################################

process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string("ewkZMuMuGolden.root")
)


zSelection = cms.PSet(
    cut = cms.string("charge = 0 & daughter(0).pt > 20 & daughter(1).pt > 20 & abs(daughter(0).eta)<2.0 & abs(daughter(1).eta)<2.0 & mass > 20"),
    isoCut = cms.double(3.),
    deltaRTrk = cms.double(0.3),
    ptThreshold = cms.double("1.5"), 
    deltaRVetoTrk = cms.double("0.015"), 
    muonIsolations = cms.InputTag("AODMuonIsoDepositCtfTk")
    )




#ZMuMu: at least one HLT trigger match
process.goodZToMuMuAtLeast1HLT = cms.EDFilter(
    "ZGoldenSelectorAndFilter",
    zSelection,
    TrigTag = cms.InputTag("TriggerResults::HLT"),
    triggerEvent = cms.InputTag( "hltTriggerSummaryAOD::HLT" ),
    src = cms.InputTag("dimuonsGlobalAOD"),
    condition =cms.string("atLeastOneMatched"),
    hltPath = cms.string("HLT_Mu9"),
    L3FilterName= cms.string("hltSingleMu9L3Filtered9"),
    maxDPtRel = cms.double( 1.0 ),
    maxDeltaR = cms.double( 0.2 ),
    filter = cms.bool(True) 
)






zPlots = cms.PSet(
    histograms = cms.VPSet(
    cms.PSet(
    min = cms.untracked.double(0.0),
    max = cms.untracked.double(200.0),
    nbins = cms.untracked.int32(200),
    name = cms.untracked.string("zMass"),
    description = cms.untracked.string("Z mass [GeV/c^{2}]"),
    plotquantity = cms.untracked.string("mass")
    ),
    cms.PSet(
    min = cms.untracked.double(0.0),
    max = cms.untracked.double(200.0),
    nbins = cms.untracked.int32(200),
    name = cms.untracked.string("mu1Pt"),
    description = cms.untracked.string("Highest muon p_{t} [GeV/c]"),
    plotquantity = cms.untracked.string("max(daughter(0).pt,daughter(1).pt)")
    ),
    cms.PSet(
    min = cms.untracked.double(0.0),
    max = cms.untracked.double(200.0),
    nbins = cms.untracked.int32(200),
    name = cms.untracked.string("mu2Pt"),
    description = cms.untracked.string("Lowest muon p_{t} [GeV/c]"),
    plotquantity = cms.untracked.string("min(daughter(0).pt,daughter(1).pt)")
    )
    )
)

process.eventInfo = cms.OutputModule (
    "AsciiOutputModule"
)


process.goodZToMuMuPlots = cms.EDFilter(
    "CandViewHistoAnalyzer",
    zPlots,
    src = cms.InputTag("goodZToMuMuAtLeast1HLT"),
    filter = cms.bool(False)
)



process.ewkZMuMuGoldenPath = cms.Path(
    process.goodAODMuons *
    process.goodAODGlobalMuons *
    process.dimuonsHLTFilter *  
    process.dimuonsAOD *
    process.dimuonsFilter *
    process.dimuonsGlobalAOD * 
    process.AODMuonIsoDepositCtfTk *
    process.goodZToMuMuAtLeast1HLT *
    process.goodZToMuMuPlots    
)



process.endPath = cms.EndPath( 
    process.eventInfo 
)
