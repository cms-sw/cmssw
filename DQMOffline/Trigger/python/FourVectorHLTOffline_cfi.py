import FWCore.ParameterSet.Config as cms

# $Id: FourVectorHLTOffline_cfi.py,v 1.5 2008/08/18 22:01:02 berryhil Exp $
hltResults = cms.EDFilter("FourVectorHLTOffline",
    plotAll = cms.untracked.bool(False),
    ptMax = cms.untracked.double(100.0),
    ptMin = cms.untracked.double(0.0),
    paths = cms.VPSet(
# single jet triggers
             cms.PSet(
              pathname = cms.string("HLTJet30"),
              filtername = cms.InputTag("hlt1jet30","","HLT"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              pathname = cms.string("HLTJet50"),
              filtername = cms.InputTag("hlt1jet50","","HLT"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              pathname = cms.string("HLTJet80"),
              filtername = cms.InputTag("hlt1jet80","","HLT"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              pathname = cms.string("HLTJet110"),
              filtername = cms.InputTag("hlt1jet110","","HLT"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              pathname = cms.string("HLTJet180"),
              filtername = cms.InputTag("hlt1jet180regional","","HLT"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              pathname = cms.string("HLTJet250"),
              filtername = cms.InputTag("hlt1jet250","","HLT"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(200.0)
             ),
#single electron triggers
             cms.PSet(
              pathname = cms.string("HLT_IsoEle15_L1I"),
              filtername = cms.InputTag("hltL1IsoSingleElectronTrackIsolFilter","","HLT"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              pathname = cms.string("HLT_IsoEle18_L1R"),
              filtername = cms.InputTag("hltL1NonIsoSingleElectronTrackIsolFilter","","HLT"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              pathname = cms.string("HLT_IsoEle18_L1R"),
              filtername = cms.InputTag("hltL1NonIsoSingleElectronTrackIsolFilter","","HLT"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              pathname = cms.string("HLT_EM80"),
              filtername = cms.InputTag("hltL1NonIsoSingleEMHighEtTrackIsolFilter","","HLT"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              pathname = cms.string("HLT_EM200"),
              filtername = cms.InputTag("hltL1NonIsoSinglePhotonEMVeryHighEtEtFilter","","HLT"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
#photon triggers
             cms.PSet(
              pathname = cms.string("HLT_IsoPhoton30_L1I"),
              filtername = cms.InputTag("hltL1IsoSinglePhotonTrackIsolFilter","","HLT"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              pathname = cms.string("HLT_IsoPhoton10_L1R"),
              filtername = cms.InputTag("hltL1NonIsoSinglePhotonEt10TrackIsolFilter","","HLT"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              pathname = cms.string("HLT_IsoPhoton40_L1R"),
              filtername = cms.InputTag("hltL1NonIsoSinglePhotonTrackIsolFilter","","HLT"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
#muon triggers
             cms.PSet(
              pathname = cms.string("HLT_L1Mu"),
              filtername = cms.InputTag("hltMuLevel1PathL1Filtered","","HLT"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              pathname = cms.string("HLT_L1MuOpen"),
              filtername = cms.InputTag("hltMuLevel1PathL1OpenFiltered","","HLT"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              pathname = cms.string("HLT_NoTrackerIsoMu15"),
              filtername = cms.InputTag("hltSingleMuNoIsoL3TkPreFilter","","HLT"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              pathname = cms.string("HLT_Mu15"),
              filtername = cms.InputTag("hltSingleMuNoIsoL3PreFiltered15","","HLT"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             )
             ),
                          
    # this is I think MC
    #triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT")
    # this is data (CRUZET I or II best guess)
    triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","FU"),
    triggerResultsLabel = cms.InputTag("TriggerResults","","FU")
)



