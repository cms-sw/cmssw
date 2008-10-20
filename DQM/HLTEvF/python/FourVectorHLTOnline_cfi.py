import FWCore.ParameterSet.Config as cms

# $Id: FourVectorHLTOnline_cfi.py,v 1.7 2008/10/02 18:44:46 berryhil Exp $
hltResultsOn = cms.EDFilter("FourVectorHLTOnline",
    plotAll = cms.untracked.bool(True),
    ptMax = cms.untracked.double(100.0),
    ptMin = cms.untracked.double(0.0),
    paths = cms.VPSet(
# single jet triggers
             cms.PSet(
              pathname = cms.string("HLTJet30"),
              filtername = cms.string("hlt1jet30"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              pathname = cms.string("HLTJet50"),
              filtername = cms.string("hlt1jet50"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              pathname = cms.string("HLTJet80"),
              filtername = cms.string("hlt1jet80"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              pathname = cms.string("HLTJet110"),
              filtername = cms.string("hlt1jet110"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              pathname = cms.string("HLTJet180"),
              filtername = cms.string("hlt1jet180regional"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              pathname = cms.string("HLTJet250"),
              filtername = cms.string("hlt1jet250"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(200.0)
             ),
#single electron triggers
             cms.PSet(
              pathname = cms.string("HLT_IsoEle15_L1I"),
              filtername = cms.string("hltL1IsoSingleElectronTrackIsolFilter"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              pathname = cms.string("HLT_IsoEle18_L1R"),
              filtername = cms.string("hltL1NonIsoSingleElectronTrackIsolFilter"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              pathname = cms.string("HLT_IsoEle18_L1R"),
              filtername = cms.string("hltL1NonIsoSingleElectronTrackIsolFilter"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              pathname = cms.string("HLT_EM80"),
              filtername = cms.string("hltL1NonIsoSingleEMHighEtTrackIsolFilter"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              pathname = cms.string("HLT_EM200"),
              filtername = cms.string("hltL1NonIsoSinglePhotonEMVeryHighEtEtFilter"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
#photon triggers
             cms.PSet(
              pathname = cms.string("HLT_IsoPhoton30_L1I"),
              filtername = cms.string("hltL1IsoSinglePhotonTrackIsolFilter"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              pathname = cms.string("HLT_IsoPhoton10_L1R"),
              filtername = cms.string("hltL1NonIsoSinglePhotonEt10TrackIsolFilter"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              pathname = cms.string("HLT_IsoPhoton40_L1R"),
              filtername = cms.string("hltL1NonIsoSinglePhotonTrackIsolFilter"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
#muon triggers
             cms.PSet(
              pathname = cms.string("HLT_L1Mu"),
              filtername = cms.string("hltMuLevel1PathL1Filtered"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              pathname = cms.string("HLT_L1MuOpen"),
              filtername = cms.string("hltMuLevel1PathL1OpenFiltered"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              pathname = cms.string("HLT_NoTrackerIsoMu15"),
              filtername = cms.string("hltSingleMuNoIsoL3TkPreFilter"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              pathname = cms.string("HLT_Mu15"),
              filtername = cms.string("hltSingleMuNoIsoL3PreFiltered15"),
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



