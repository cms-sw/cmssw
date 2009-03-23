# AlCaReco for muon based alignment using ZMuMu events
import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOMuAlZMuMuHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    eventSetupPathsKey = 'MuAlZMuMu',
    throw = False # tolerate triggers not available
    )

import Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi
ALCARECOMuAlZMuMu = Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi.AlignmentMuonSelector.clone(
    applyMassPairFilter = True,
    minMassPair = 91. - 10.,
    maxMassPair = 91. + 10.
    )

seqALCARECOMuAlZMuMu = cms.Sequence(ALCARECOMuAlZMuMuHLT+ALCARECOMuAlZMuMu)

