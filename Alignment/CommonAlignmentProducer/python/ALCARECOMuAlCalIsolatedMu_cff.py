# AlCaReco for muon based alignment using any individual muon tracks
import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOMuAlCalIsolatedMuHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    HLTPaths = ["HLT_L1MuOpen", "HLT_L1Mu", "HLT_L2Mu9",
                "HLT_Mu3", "HLT_Mu5", "HLT_Mu9", "HLT_Mu11",
                "HLT_DoubleMu3"],
    throw = False # tolerate triggers stated above, but not available
    )

import Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi
ALCARECOMuAlCalIsolatedMu = Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi.AlignmentMuonSelector.clone(
    src = cms.InputTag("muons"),
    filter = True, # not strictly necessary, but provided for symmetry with MuAlStandAloneCosmics
# DT calibration needs as many muons with DIGIs as possible, which in cosmic ray runs means standAloneMuons
#    nHitMinGB = 1, # muon collections now merge globalMuons, standAlone, and trackerMuons: this stream has always assumed globalMuons
    )

seqALCARECOMuAlCalIsolatedMu = cms.Sequence(ALCARECOMuAlCalIsolatedMuHLT + ALCARECOMuAlCalIsolatedMu)
