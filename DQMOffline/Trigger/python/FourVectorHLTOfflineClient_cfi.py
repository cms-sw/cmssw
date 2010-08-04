# $Id: FourVectorHLTOfflineClient_cfi.py,v 1.14 2010/06/22 18:33:13 rekovic Exp $
import FWCore.ParameterSet.Config as cms

hltFourVectorClient = cms.EDAnalyzer("FourVectorHLTClient",
    hltClientDir = cms.untracked.string('HLT/FourVector/paths/'),
    hltSourceDir = cms.untracked.string('HLT/FourVector/paths/'),
    prescaleLS = cms.untracked.int32(-1),
    prescaleEvt = cms.untracked.int32(1),
    processname = cms.untracked.string("HLT"),
    customEffDir = cms.untracked.string('custom-eff'),
    effpaths = cms.VPSet(
             cms.PSet(
              pathname = cms.string("HLT_"),
              denompathname = cms.string("")  
             ),
             cms.PSet(
              pathname = cms.string("HLT_"),
              denompathname = cms.string("MinBias")  
             ),
             cms.PSet(
              pathname = cms.string("EG"),
              denompathname = cms.string("HLT_Mu")  
             ),
             cms.PSet(
              pathname = cms.string("EG"),
              denompathname = cms.string("HLT_Mu7")  
             ),
             cms.PSet(
              pathname = cms.string("Jet"),
              denompathname = cms.string("HLT_Mu")  
             ),
             cms.PSet(
              pathname = cms.string("Jet"),
              denompathname = cms.string("HLT_Mu7")  
             ),
             cms.PSet(
              pathname = cms.string("Ele"),
              denompathname = cms.string("HLT_Mu")  
             ),
             cms.PSet(
              pathname = cms.string("Ele"),
              denompathname = cms.string("HLT_Mu7")  
             ),
             cms.PSet(
              pathname = cms.string("Pho"),
              denompathname = cms.string("HLT_Mu")  
             ),
             cms.PSet(
              pathname = cms.string("Pho"),
              denompathname = cms.string("HLT_Mu7")  
             ),
             cms.PSet(
              pathname = cms.string("Tau"),
              denompathname = cms.string("HLT_Mu7")  
             ),
             cms.PSet(
              pathname = cms.string("MET"),
              denompathname = cms.string("HLT_Mu7")  
             ),
             cms.PSet(
              pathname = cms.string("Mu"),
              denompathname = cms.string("HLT_Jet")  
             ),
             cms.PSet(
              pathname = cms.string("Mu"),
              denompathname = cms.string("HLT_Jet50U")  
             )

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

