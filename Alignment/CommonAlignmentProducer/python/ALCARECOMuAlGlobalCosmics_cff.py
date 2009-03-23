# AlCaReco for muon alignment using global cosmic ray tracks
import FWCore.ParameterSet.Config as cms

# HLT
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOMuAlGlobalCosmicsHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    eventSetupPathsKey = 'MuAlGlobalCosmics',
    throw = False # tolerate triggers not available
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
