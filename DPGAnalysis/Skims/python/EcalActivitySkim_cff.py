import FWCore.ParameterSet.Config as cms

HLTPath = "HLT_Activity_Ecal*"

import HLTrigger.HLTfilters.hltHighLevel_cfi
ecalActivityHltFilter = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    throw = cms.bool(False),
    HLTPaths = [HLTPath]
    )

ecalActivitySeq = cms.Sequence( ecalActivityHltFilter )
