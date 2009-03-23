# AlCaReco for muon alignment using stand-alone cosmic ray tracks
import FWCore.ParameterSet.Config as cms

# HLT
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOMuAlStandAloneCosmicsHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    eventSetupPathsKey = 'MuAlStandAloneCosmics',
    throw = False # tolerate triggers not available
    )
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi

ALCARECOMuAlStandAloneCosmics = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone(
    src = "cosmicMuons", #cms.InputTag("cosmicMuons")
    filter = True,
    ptMin = 10.0,
    etaMin = -100.0,
    etaMax = 100.0
    )

seqALCARECOMuAlStandAloneCosmics = cms.Sequence(ALCARECOMuAlStandAloneCosmicsHLT + ALCARECOMuAlStandAloneCosmics)

