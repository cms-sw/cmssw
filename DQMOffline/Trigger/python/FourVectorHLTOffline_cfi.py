import FWCore.ParameterSet.Config as cms

# $Id: FourVectorHLTOffline_cfi.py,v 1.8 2008/10/21 22:47:15 berryhil Exp $
hltResults = cms.EDFilter("FourVectorHLTOffline",
    plotAll = cms.untracked.bool(True),
    ptMax = cms.untracked.double(100.0),
    ptMin = cms.untracked.double(0.0),
    paths = cms.VPSet(
# single jet triggers
             cms.PSet(
              denompathname = cms.string("HLTJet30"),  
              pathname = cms.string("HLTJet30"),
              l1pathname = cms.string("dummy"),  
              filtername = cms.string("hlt1jet30"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              denompathname = cms.string("HLTJet50"),
              pathname = cms.string("HLTJet50"),
              l1pathname = cms.string("dummy"),  
              filtername = cms.string("hlt1jet50"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              denompathname = cms.string("HLTJet80"),
              pathname = cms.string("HLTJet80"),
              l1pathname = cms.string("dummy"),  
              filtername = cms.string("hlt1jet80"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              denompathname = cms.string("HLTJet110"),
              pathname = cms.string("HLTJet110"),
              l1pathname = cms.string("dummy"),  
              filtername = cms.string("hlt1jet110"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              denompathname = cms.string("HLTJet180"),
              pathname = cms.string("HLTJet180"),
              l1pathname = cms.string("dummy"),  
              filtername = cms.string("hlt1jet180regional"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              denompathname = cms.string("HLTJet250"),
              pathname = cms.string("HLTJet250"),
              l1pathname = cms.string("dummy"),  
              filtername = cms.string("hlt1jet250"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(200.0)
             ),
#single electron triggers
             cms.PSet(
              denompathname = cms.string("HLT_IsoEle15_L1I"),
              pathname = cms.string("HLT_IsoEle15_L1I"),
              l1pathname = cms.string("dummy"),  
              filtername = cms.string("hltL1IsoSingleElectronTrackIsolFilter"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              denompathname = cms.string("HLT_IsoEle18_L1R"),
              pathname = cms.string("HLT_IsoEle18_L1R"),
              l1pathname = cms.string("dummy"),  
              filtername = cms.string("hltL1NonIsoSingleElectronTrackIsolFilter"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              denompathname = cms.string("HLT_IsoEle18_L1R"),
              pathname = cms.string("HLT_IsoEle18_L1R"),
              l1pathname = cms.string("dummy"),  
              filtername = cms.string("hltL1NonIsoSingleElectronTrackIsolFilter"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              denompathname = cms.string("HLT_EM80"),
              pathname = cms.string("HLT_EM80"),
              l1pathname = cms.string("dummy"),  
              filtername = cms.string("hltL1NonIsoSingleEMHighEtTrackIsolFilter"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              denompathname = cms.string("HLT_EM200"),
              pathname = cms.string("HLT_EM200"),
              l1pathname = cms.string("dummy"),  
              filtername = cms.string("hltL1NonIsoSinglePhotonEMVeryHighEtEtFilter"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
#photon triggers
             cms.PSet(
              denompathname = cms.string("HLT_IsoPhoton30_L1I"),
              pathname = cms.string("HLT_IsoPhoton30_L1I"),
              l1pathname = cms.string("dummy"),  
              filtername = cms.string("hltL1IsoSinglePhotonTrackIsolFilter"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              denompathname = cms.string("HLT_IsoPhoton10_L1R"),
              pathname = cms.string("HLT_IsoPhoton10_L1R"),
              l1pathname = cms.string("dummy"),  
              filtername = cms.string("hltL1NonIsoSinglePhotonEt10TrackIsolFilter"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              denompathname = cms.string("HLT_IsoPhoton40_L1R"),
              pathname = cms.string("HLT_IsoPhoton40_L1R"),
              l1pathname = cms.string("dummy"),  
              filtername = cms.string("hltL1NonIsoSinglePhotonTrackIsolFilter"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
#muon triggers
             cms.PSet(
              denompathname = cms.string("HLT_L1Mu"),
              pathname = cms.string("HLT_L1Mu"),
              l1pathname = cms.string("dummy"),  
              filtername = cms.string("hltMuLevel1PathL1Filtered"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              denompathname = cms.string("HLT_L1MuOpen"),
              pathname = cms.string("HLT_L1MuOpen"),
              l1pathname = cms.string("dummy"),  
              filtername = cms.string("hltMuLevel1PathL1OpenFiltered"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              denompathname = cms.string("HLT_NoTrackerIsoMu15"),
              pathname = cms.string("HLT_NoTrackerIsoMu15"),
              l1pathname = cms.string("dummy"),  
              filtername = cms.string("hltSingleMuNoIsoL3TkPreFilter"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             ),
             cms.PSet(
              denompathname = cms.string("HLT_Mu15"),
              pathname = cms.string("HLT_Mu15"),
              l1pathname = cms.string("dummy"),  
              filtername = cms.string("hltSingleMuNoIsoL3PreFiltered15"),
              type = cms.uint32(0),
              ptMin = cms.untracked.double(0.0),
              ptMax = cms.untracked.double(100.0)
             )
             ),
                          
     # this is I think MC and CRUZET4
    triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    triggerResultsLabel = cms.InputTag("TriggerResults","","HLT"),
    processname = cms.string("HLT")

    # this is data (CRUZET I or II best guess)
    #triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","FU"),
    #triggerResultsLabel = cms.InputTag("TriggerResults","","FU"),
    #processname = cms.string("FU")

 )
