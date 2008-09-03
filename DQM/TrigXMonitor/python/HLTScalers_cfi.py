import FWCore.ParameterSet.Config as cms

# HLT scalers. wittich 11/07
# $Id: HLTScalers_cfi.py,v 1.5 2008/09/02 02:37:22 wittich Exp $
hlts = cms.EDFilter("HLTScalers",
    #    untracked bool specifyPaths = true
    #    untracked vstring pathNames = {'HLT1MuonIso',
    #  				'HLT1MuonNonIso',
    #  				'HLT2MuonIso',
    #  				'HLT2MuonNonIso'}
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    dqmFolder = cms.untracked.string("HLT/HLTScalers_EvF"),
    verbose = cms.untracked.bool(False)
)

