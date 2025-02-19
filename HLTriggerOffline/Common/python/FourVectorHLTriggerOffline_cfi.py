# $Id: FourVectorHLTriggerOffline_cfi.py,v 1.14 2010/02/17 17:47:41 wmtan Exp $

import FWCore.ParameterSet.Config as cms

hltriggerResults = cms.EDAnalyzer("FourVectorHLTriggerOffline",
    dirname = cms.untracked.string("HLT/FourVector_Val/source"),
    plotAll = cms.untracked.bool(False),
    ptMax = cms.untracked.double(100.0),
    ptMin = cms.untracked.double(0.0),

  electronEtaMax = cms.untracked.double(2.5),
  electronEtMin = cms.untracked.double(3.0),
  electronDRMatch  =cms.untracked.double(0.5),

  muonEtaMax = cms.untracked.double(2.1),
  muonEtMin = cms.untracked.double(3.0),
  muonDRMatch  =cms.untracked.double(0.3),

  tauEtaMax = cms.untracked.double(5.0),
  tauEtMin = cms.untracked.double(3.0),
  tauDRMatch  =cms.untracked.double(0.5),

  jetEtaMax = cms.untracked.double(5.0),
  jetEtMin = cms.untracked.double(10.0),
  jetDRMatch  =cms.untracked.double(0.3),

  bjetEtaMax = cms.untracked.double(2.5),
  bjetEtMin = cms.untracked.double(10.0),
  bjetDRMatch  =cms.untracked.double(0.3),

  photonEtaMax = cms.untracked.double(2.5),
  photonEtMin = cms.untracked.double(3.0),
  photonDRMatch  =cms.untracked.double(0.5),

  trackEtaMax = cms.untracked.double(2.5),
  trackEtMin = cms.untracked.double(3.0),
  trackDRMatch  =cms.untracked.double(0.3),

  metMin = cms.untracked.double(10.0),
  htMin = cms.untracked.double(10.0),
  sumEtMin = cms.untracked.double(10.0),

    paths = cms.VPSet(
            # cms.PSet(
            #  pathname = cms.string("HLT_Mu11"),
            #  denompathname = cms.string(""),  
            # ),
               #cms.PSet(
               # pathname = cms.string("HLT_Ele10_SW_L1R"),
               # denompathname = cms.string(""),  
               #)
#             cms.PSet(
#              pathname = cms.string("HLT_Photon15_L1R"),
#              denompathname = cms.string(""),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_MET25"),
#              denompathname = cms.string(""),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_HT250"),
#              denompathname = cms.string(""),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_BTagMu_Jet20"),
#              denompathname = cms.string(""),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet30"),
#              denompathname = cms.string(""),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Mu11"),
#              denompathname = cms.string("HLT_L1Jet15"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet30"),
#              denompathname = cms.string("HLT_Mu3"),  
#             )
#             cms.PSet(
#              pathname = cms.string("HLT_IsoEle15_L1I"),
#              denompathname = cms.string("HLT_IsoEle15_LW_L1I"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Ele15_SW_L1R"),
#              denompathname = cms.string("HLT_Ele15_SW_L1R"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Ele15_LW_L1R"),
#              denompathname = cms.string("HLT_Ele10_SW_L1R"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Ele15_LW_L1R"),
#              denompathname = cms.string("HLT_Ele15_LW_L1R"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Mu3"),
#              denompathname = cms.string("HLT_L1Jet15"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Mu3"),
#              denompathname = cms.string("HLT_Jet30"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Mu3"),
#              denompathname = cms.string("HLT_L1Mu"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Mu3"),
#              denompathname = cms.string("HLT_Mu3"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Mu11"),
#              denompathname = cms.string("HLT_L1Mu"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Mu11"),
#              denompathname = cms.string("HLT_Mu11"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Mu11"),
#              denompathname = cms.string("HLT_L1Jet15"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet30"),
#              denompathname = cms.string("HLT_L1Mu"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet30"),
#              denompathname = cms.string("HLT_L1Jet15"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet30"),
#              denompathname = cms.string("HLT_L1Jet30"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet50"),
#              denompathname = cms.string("HLT_Jet30"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet50"),
#              denompathname = cms.string("HLT_Jet50"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet80"),
#              denompathname = cms.string("HLT_Jet30"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet80"),
#              denompathname = cms.string("HLT_Jet80"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet110"),
#              denompathname = cms.string("HLT_Jet50"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet80"),
#              denompathname = cms.string("HLT_L1Mu"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet110"),
#              denompathname = cms.string("HLT_Jet110"),  
#             )
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
