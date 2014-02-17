## $Id: FourVectorHLTriggerOfflineClient_cfi.py,v 1.12 2010/03/01 21:25:07 wmtan Exp $

import FWCore.ParameterSet.Config as cms

hltriggerFourVectorClient = cms.EDAnalyzer("FourVectorHLTClient",
    hltClientDir = cms.untracked.string('HLT/FourVector_Val/client/'),
    hltSourceDir = cms.untracked.string('HLT/FourVector_Val/source/'),
    prescaleLS = cms.untracked.int32(-1),
    prescaleEvt = cms.untracked.int32(1),
    customEffDir = cms.untracked.string('custom-eff'),
    effpaths = cms.VPSet(
             cms.PSet(
              pathname = cms.string("HLT_"),
              denompathname = cms.string(""),  
             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Mu5"),
#              denompathname = cms.string(""),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Mu9"),
#              denompathname = cms.string(""),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Mu11"),
#              denompathname = cms.string(""),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Ele10_SW_L1R"),
#              denompathname = cms.string(""),  
#             ),
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
#              pathname = cms.string("HLT_Jet10U"),
#              denompathname = cms.string(""),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet15U"),
#              denompathname = cms.string(""),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet30U"),
#              denompathname = cms.string(""),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet50U"),
#              denompathname = cms.string(""),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet30"),
#              denompathname = cms.string(""),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet50"),
#              denompathname = cms.string(""),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet80"),
#              denompathname = cms.string(""),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet110"),
#              denompathname = cms.string(""),  
#             ),
             cms.PSet(
              pathname = cms.string("HLT_Mu9"),
              denompathname = cms.string("HLT_Jet30"),  
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
#              pathname = cms.string("HLT_Jet80"),
#              denompathname = cms.string("HLT_L1Mu"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet110"),
#              denompathname = cms.string("HLT_Jet50"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet110"),
#              denompathname = cms.string("HLT_Jet110"),  
#             )
    )

)

