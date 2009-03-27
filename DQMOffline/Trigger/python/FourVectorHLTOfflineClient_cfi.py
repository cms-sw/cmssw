import FWCore.ParameterSet.Config as cms

hltFourVectorClient = cms.EDFilter("FourVectorHLTClient",
    hltClientDir = cms.untracked.string('HLTOffline/FourVectorHLTOfflineClient/'),
    hltSourceDir = cms.untracked.string('HLTOffline/FourVectorHLTOfflinehltResults'),
    prescaleLS = cms.untracked.int32(-1),
    prescaleEvt = cms.untracked.int32(1),
    customEffDir = cms.untracked.string('custom-eff'),
    effpaths = cms.VPSet(
             cms.PSet(
              pathname = cms.string("HLT_Mu11"),
              denompathname = cms.string("HLT_L1Jet15"),  
             ),
             cms.PSet(
              pathname = cms.string("HLT_Jet30"),
              denompathname = cms.string("HLT_Mu3"),  
             )
#             cms.PSet(
#              pathname = cms.string("HLT_L2Mu3"),
#              denompathname = cms.string("HLT_L1Mu"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet30"),
#              denompathname = cms.string("HLT_L1Jet15"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet50"),
#              denompathname = cms.string("HLT_Jet30"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet110"),
#              denompathname = cms.string("HLT_Jet50"),  
#             )
    )

)

