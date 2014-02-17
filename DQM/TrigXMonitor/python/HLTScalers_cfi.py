import FWCore.ParameterSet.Config as cms

# HLT scalers. wittich 11/07
# $Id: HLTScalers_cfi.py,v 1.8 2011/03/24 18:25:45 rekovic Exp $
hlts = cms.EDAnalyzer("HLTScalers",
    #    untracked bool specifyPaths = true
    #    untracked vstring pathNames = {'HLT1MuonIso',
    #  				'HLT1MuonNonIso',
    #  				'HLT2MuonIso',
    #  				'HLT2MuonNonIso'}
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    processname = cms.string("HLT"),
    dqmFolder = cms.untracked.string("HLT/HLTScalers_EvF"),
    verbose = cms.untracked.bool(False)
)

