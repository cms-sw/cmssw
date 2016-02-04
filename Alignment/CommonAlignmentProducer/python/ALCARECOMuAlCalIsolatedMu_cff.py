# AlCaReco for muon based alignment using any individual muon tracks
import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOMuAlCalIsolatedMuHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    eventSetupPathsKey = 'MuAlCalIsolatedMu',
    throw = False # tolerate triggers not available
    )

import Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi
ALCARECOMuAlCalIsolatedMu = Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi.AlignmentMuonSelector.clone(
    src = cms.InputTag("muons"),
    filter = True, # not strictly necessary, but provided for symmetry with MuAlStandAloneCosmics
# DT calibration needs as many muons with DIGIs as possible, which in cosmic ray runs means standAloneMuons
#    nHitMinGB = 1, # muon collections now merge globalMuons, standAlone, and trackerMuons: this stream has always assumed globalMuons
    ptMin=cms.double(0.),
    pMin=cms.double(10.)
    )

seqALCARECOMuAlCalIsolatedMu = cms.Sequence(ALCARECOMuAlCalIsolatedMuHLT + ALCARECOMuAlCalIsolatedMu)
