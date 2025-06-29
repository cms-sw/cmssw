# AlCaReco for track based alignment using HLT tracks
import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi

ALCARECOTkAlHLTTracksHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    # eventSetupPathsKey = 'TkAlMinBias',
    HLTPaths = ['HLT_*'],
    throw = False # tolerate triggers stated above, but not available
    )

# DCS partitions
# "EBp","EBm","EEp","EEm","HBHEa","HBHEb","HBHEc","HF","HO","RPC"
# "DT0","DTp","DTm","CSCp","CSCm","CASTOR","TIBTID","TOB","TECp","TECm"
# "BPIX","FPIX","ESp","ESm"
import DPGAnalysis.Skims.skim_detstatus_cfi
ALCARECOTkAlHLTTracksDCSFilter = DPGAnalysis.Skims.skim_detstatus_cfi.dcsstatus.clone(
    DetectorType = cms.vstring('TIBTID','TOB','TECp','TECm','BPIX','FPIX'),
    ApplyFilter  = cms.bool(True),
    AndOr        = cms.bool(True),
    DebugOn      = cms.untracked.bool(False)
)

import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOTkAlHLTTracks = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()
ALCARECOTkAlHLTTracks.src = cms.InputTag("hltMergedTracks") # run on hltMergedTracks instead of generalTracks
ALCARECOTkAlHLTTracks.filter = True ##do not store empty events	

ALCARECOTkAlHLTTracks.applyBasicCuts = True
ALCARECOTkAlHLTTracks.ptMin = 0.65 ##GeV
ALCARECOTkAlHLTTracks.pMin = 1.5 ##GeV

ALCARECOTkAlHLTTracks.etaMin = -3.5
ALCARECOTkAlHLTTracks.etaMax = 3.5
ALCARECOTkAlHLTTracks.nHitMin = 7 ## at least 7 hits required
ALCARECOTkAlHLTTracks.GlobalSelector.applyIsolationtest = False
ALCARECOTkAlHLTTracks.GlobalSelector.applyGlobalMuonFilter = False
ALCARECOTkAlHLTTracks.TwoBodyDecaySelector.applyMassrangeFilter = False
ALCARECOTkAlHLTTracks.TwoBodyDecaySelector.applyChargeFilter = False
ALCARECOTkAlHLTTracks.TwoBodyDecaySelector.applyAcoplanarityFilter = False

seqALCARECOTkAlHLTTracks = cms.Sequence(ALCARECOTkAlHLTTracksHLT+ALCARECOTkAlHLTTracksDCSFilter+ALCARECOTkAlHLTTracks)
