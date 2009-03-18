# AlCaReco for muon based alignment using ZMuMu events
import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOMuAlZMuMuHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    #HLTPaths = ['HLT2MuonIso', 'HLT2MuonNonIso', 'HLT2MuonZ']
    HLTPaths = ['HLT_DoubleIsoMu3', 'HLT_DoubleMu3', 'HLT_DoubleMu7_Z'],
    throw = False # tolerate triggers stated above, but not available
    )

import Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi
ALCARECOMuAlZMuMu = Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi.AlignmentMuonSelector.clone()
ALCARECOMuAlZMuMu.applyMassPairFilter = cms.bool(True)
ALCARECOMuAlZMuMu.minMassPair = cms.double(91. - 10.)
ALCARECOMuAlZMuMu.maxMassPair = cms.double(91. + 10.)

seqALCARECOMuAlZMuMu = cms.Sequence(ALCARECOMuAlZMuMuHLT+ALCARECOMuAlZMuMu)

