# AlCaReco for muon based alignment using any individual muon tracks
import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOMuAlCalIsolatedMuHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    HLTPaths = ["HLT_L1MuOpen", "HLT_L1Mu",
                "HLT_Mu3", "HLT_Mu5", "HLT_Mu7",
                "HLT_Mu9", "HLT_Mu11", "HLT_Mu13",
                "HLT_Mu15", "HLT_L2Mu9",
                "HLT_IsoMu9", "HLT_IsoMu11",
                "HLT_IsoMu13", "HLT_IsoMu15"]
    )

import Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi
ALCARECOMuAlCalIsolatedMu = Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi.AlignmentMuonSelector.clone()

seqALCARECOMuAlCalIsolatedMu = cms.Sequence(ALCARECOMuAlCalIsolatedMuHLT + ALCARECOMuAlCalIsolatedMu)
