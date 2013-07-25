# $Id: FourVectorHLTOfflineClient_cfi.py,v 1.16 2010/08/04 14:27:51 rekovic Exp $
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
              denompathname = cms.string("HLT_Mu3")  
             ),
             cms.PSet(
              pathname = cms.string("EG"),
              denompathname = cms.string("HLT_Mu5")  
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
              denompathname = cms.string("HLT_Mu3")  
             ),
             cms.PSet(
              pathname = cms.string("Jet"),
              denompathname = cms.string("HLT_Mu5")  
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
              denompathname = cms.string("HLT_Mu3")  
             ),
             cms.PSet(
              pathname = cms.string("Ele"),
              denompathname = cms.string("HLT_Mu5")  
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
              denompathname = cms.string("HLT_Mu3")  
             ),
             cms.PSet(
              pathname = cms.string("Pho"),
              denompathname = cms.string("HLT_Mu5")  
             ),
             cms.PSet(
              pathname = cms.string("Pho"),
              denompathname = cms.string("HLT_Mu7")  
             ),
             cms.PSet(
              pathname = cms.string("Tau"),
              denompathname = cms.string("HLT_Mu")  
             ),
             cms.PSet(
              pathname = cms.string("MET"),
              denompathname = cms.string("HLT_Mu")  
             ),
             cms.PSet(
              pathname = cms.string("Mu"),
              denompathname = cms.string("HLT_Jet")  
             ),
             cms.PSet(
              pathname = cms.string("Mu"),
              denompathname = cms.string("HLT_Jet15U")  
             ),
             cms.PSet(
              pathname = cms.string("Mu"),
              denompathname = cms.string("HLT_Jet30U")  
             ),
             cms.PSet(
              pathname = cms.string("Mu"),
              denompathname = cms.string("HLT_Jet50U")  
             )

    )

)

