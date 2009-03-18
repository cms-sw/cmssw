# AlCaReco for muon alignment using global cosmic ray tracks
import FWCore.ParameterSet.Config as cms

# HLT
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOMuAlGlobalCosmicsHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    HLTPaths = ["HLT_L1MuOpen", "HLT_L1Mu", "HLT_L2Mu9",
                "HLT_Mu3", "HLT_Mu5", "HLT_Mu9", "HLT_Mu11",
                "HLT_DoubleMu3", "HLT_TrackerCosmics"],
    throw = False # tolerate triggers stated above, but not available
    )

import Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi

ALCARECOMuAlGlobalCosmics = Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi.AlignmentMuonSelector.clone(
    src = cms.InputTag("muonsBarrelOnly"),
    filter = True, # not strictly necessary, but provided for symmetry with MuAlStandAloneCosmics
    nHitMinGB = 1,
    ptMin = 10.0,
    etaMin = -100.0,
    etaMax =  100.0
    )

seqALCARECOMuAlGlobalCosmics = cms.Sequence(ALCARECOMuAlGlobalCosmicsHLT + ALCARECOMuAlGlobalCosmics)
