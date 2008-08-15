import FWCore.ParameterSet.Config as cms

# HLT scalers. wittich 11/07
# $Id: HLTScalers.cfi,v 1.4 2008/08/01 14:38:11 bjbloom Exp $
# $Log: HLTScalers.cfi,v $
# Revision 1.4  2008/08/01 14:38:11  bjbloom
# Example of specifying cross-correlated paths
#
# Revision 1.3  2007/12/04 20:24:32  wittich
# - make hlt histograms variable width depending on path
# - add strings for path names
# - add int for nprocessed
# - add L1 scaler locally derived on Kaori's suggestion
#   + updates to cfi file for this, need to include unpacking of GT
#
# Revision 1.2  2007/12/01 19:28:51  wittich
# - fix cfi file (debug -> verbose, HLT -> FU for TriggerResults  label)
# - handle multiple beginRun for same run (don't call reset on DQM )
# - remove PathTimerService from cfg file in test subdir
#
hlts = cms.EDFilter("HLTScalers",
    #    untracked bool specifyPaths = true
    #    untracked vstring pathNames = {'HLT1MuonIso',
    #  				'HLT1MuonNonIso',
    #  				'HLT2MuonIso',
    #  				'HLT2MuonNonIso'}
    triggerResults = cms.InputTag("TriggerResults","","FU"),
    l1GtData = cms.InputTag("l1GtUnpack","","HLT"),
    verbose = cms.untracked.bool(False)
)

