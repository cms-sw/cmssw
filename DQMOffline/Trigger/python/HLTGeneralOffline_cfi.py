import FWCore.ParameterSet.Config as cms

# $Id: HLTGeneralOffline_cfi.py,v 1.2 2012/08/07 12:19:32 muell149 Exp $
hltResults = cms.EDAnalyzer("GeneralHLTOffline",
    dirname = cms.untracked.string("HLT/General/paths"),
    muonRecoCollectionName = cms.untracked.string("muons"),
    plotAll = cms.untracked.bool(False),

    ptMax = cms.untracked.double(100.0),
    ptMin = cms.untracked.double(0.0),
    Nbins = cms.untracked.uint32(50),
    Nbins2D = cms.untracked.uint32(40),
    referenceBX= cms.untracked.uint32(1),
    NLuminositySegments= cms.untracked.uint32(2000),
    LuminositySegmentSize= cms.untracked.double(23),
    NbinsOneOverEt = cms.untracked.uint32(1000),

    muonEtaMax = cms.untracked.double(2.1),

    jetEtMin = cms.untracked.double(5.0),
    jetEtaMax = cms.untracked.double(3.0),

    electronEtMin = cms.untracked.double(5.0),

    photonEtMin = cms.untracked.double(5.0),

    tauEtMin = cms.untracked.double(10.0),
                          
     # this is I think MC and CRUZET4
    triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    triggerResultsLabel = cms.InputTag("TriggerResults","","HLT"),
    HltProcessName = cms.string("HLT"),
    processname = cms.string("HLT")


 )

