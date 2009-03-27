import FWCore.ParameterSet.Config as cms

# $Id: FourVectorHLTriggerOffline_cfi.py,v 1.6 2009/02/25 23:13:38 berryhil Exp $
hltResults = cms.EDFilter("FourVectorHLTriggerOffline",
    dirname = cms.untracked.string("HLT/FourVector/source"),
    plotAll = cms.untracked.bool(False),
    ptMax = cms.untracked.double(100.0),
    ptMin = cms.untracked.double(0.0),
    paths = cms.VPSet(
             cms.PSet(
              pathname = cms.string("HLT_Mu11"),
              denompathname = cms.string("HLT_L1Jet15"),  
             ),
             cms.PSet(
              pathname = cms.string("HLT_Jet30"),
              denompathname = cms.string("HLT_Mu3"),  
             )
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
