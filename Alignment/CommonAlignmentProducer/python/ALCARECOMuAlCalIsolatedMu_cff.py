import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
# AlCaReco for muon based alignment using ZMuMu events
ALCARECOMuAlCalIsolatedMuHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
import Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi
ALCARECOMuAlCalIsolatedMu = Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi.AlignmentMuonSelector.clone()
seqALCARECOMuAlCalIsolatedMu = cms.Sequence(ALCARECOMuAlCalIsolatedMuHLT+ALCARECOMuAlCalIsolatedMu)
ALCARECOMuAlCalIsolatedMuHLT.andOr = True ## choose logical OR between Triggerbits

ALCARECOMuAlCalIsolatedMuHLT.HLTPaths = ['HLT1MuonIso', 'HLT1MuonNonIso']

