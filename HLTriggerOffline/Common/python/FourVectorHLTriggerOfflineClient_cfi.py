import FWCore.ParameterSet.Config as cms

HLTriggerOfflineFourVectorClient = cms.EDFilter("FourVectorHLTClient",
    hltClientDir = cms.untracked.string('HLT/FourVector/client/'),
    hltSourceDir = cms.untracked.string('HLT/FourVector/source/'),
    prescaleLS = cms.untracked.int32(-1),
    prescaleEvt = cms.untracked.int32(1),
    customEffDir = cms.untracked.string('custom-eff'),
    effpaths = cms.VPSet(
# single jet triggers
             cms.PSet(
              pathname = cms.string("HLT_L2Mu3"),
              denompathname = cms.string("HLT_L1Mu"),  
             ),
             cms.PSet(
              pathname = cms.string("HLT_Jet30"),
              denompathname = cms.string("HLT_L1Jet15"),  
             ),
             cms.PSet(
              pathname = cms.string("HLT_Jet50"),
              denompathname = cms.string("HLT_Jet30"),  
             ),
             cms.PSet(
              pathname = cms.string("HLT_Jet110"),
              denompathname = cms.string("HLT_Jet50"),  
             )
    )

)

