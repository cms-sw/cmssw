import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
# AlCaReco for track based alignment using ZMuMu events
ALCARECOTkAlHLTTracksZMuMuHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    # eventSetupPathsKey = 'TkAlZMuMu',
    HLTPaths = ['HLT_*Mu*'],
    throw = False # tolerate triggers stated above, but not available
    )

# DCS partitions
# "EBp","EBm","EEp","EEm","HBHEa","HBHEb","HBHEc","HF","HO","RPC"
# "DT0","DTp","DTm","CSCp","CSCm","CASTOR","TIBTID","TOB","TECp","TECm"
# "BPIX","FPIX","ESp","ESm"
import DPGAnalysis.Skims.skim_detstatus_cfi
ALCARECOTkAlHLTTracksZMuMuDCSFilter = DPGAnalysis.Skims.skim_detstatus_cfi.dcsstatus.clone(
    DetectorType = cms.vstring('TIBTID','TOB','TECp','TECm','BPIX','FPIX',
                               'DT0','DTp','DTm','CSCp','CSCm'),
    ApplyFilter  = cms.bool(True),
    AndOr        = cms.bool(True),
    DebugOn      = cms.untracked.bool(False)
)

import Alignment.CommonAlignmentProducer.TkAlMuonSelectors_cfi
ALCARECOTkAlHLTTracksZMuMuGoodMuons = Alignment.CommonAlignmentProducer.TkAlMuonSelectors_cfi.TkAlGoodIdMuonSelector.clone(
    #    src =  cms.InputTag("hltPFMuonMerging") # TODO type cast to muon ???
)
ALCARECOTkAlHLTTracksZMuMuRelCombIsoMuons = Alignment.CommonAlignmentProducer.TkAlMuonSelectors_cfi.TkAlRelCombIsoMuonSelector.clone(
    src = 'ALCARECOTkAlHLTTracksZMuMuGoodMuons'
)

import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOTkAlHLTTracksZMuMu = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()
ALCARECOTkAlHLTTracksZMuMu.src = cms.InputTag("hltMergedTracks") 
ALCARECOTkAlHLTTracksZMuMu.filter = True ##do not store empty events

ALCARECOTkAlHLTTracksZMuMu.applyBasicCuts = True
ALCARECOTkAlHLTTracksZMuMu.ptMin = 15.0 ##GeV
ALCARECOTkAlHLTTracksZMuMu.etaMin = -3.5
ALCARECOTkAlHLTTracksZMuMu.etaMax = 3.5
ALCARECOTkAlHLTTracksZMuMu.nHitMin = 0

ALCARECOTkAlHLTTracksZMuMu.GlobalSelector.muonSource = 'ALCARECOTkAlHLTTracksZMuMuRelCombIsoMuons'
# Isolation is shifted to the muon preselection, and then applied intrinsically if applyGlobalMuonFilter = True
ALCARECOTkAlHLTTracksZMuMu.GlobalSelector.applyIsolationtest = False
ALCARECOTkAlHLTTracksZMuMu.GlobalSelector.applyGlobalMuonFilter = True

ALCARECOTkAlHLTTracksZMuMu.TwoBodyDecaySelector.applyMassrangeFilter = True
ALCARECOTkAlHLTTracksZMuMu.TwoBodyDecaySelector.minXMass = 65.0 ##GeV
ALCARECOTkAlHLTTracksZMuMu.TwoBodyDecaySelector.maxXMass = 115.0 ##GeV
ALCARECOTkAlHLTTracksZMuMu.TwoBodyDecaySelector.daughterMass = 0.105 ##GeV (Muons)
ALCARECOTkAlHLTTracksZMuMu.TwoBodyDecaySelector.applyChargeFilter = True
ALCARECOTkAlHLTTracksZMuMu.TwoBodyDecaySelector.charge = 0
ALCARECOTkAlHLTTracksZMuMu.TwoBodyDecaySelector.applyAcoplanarityFilter = False
ALCARECOTkAlHLTTracksZMuMu.TwoBodyDecaySelector.numberOfCandidates = 1

seqALCARECOTkAlHLTTracksZMuMu = cms.Sequence(ALCARECOTkAlHLTTracksZMuMuHLT+ALCARECOTkAlHLTTracksZMuMuDCSFilter+ALCARECOTkAlHLTTracksZMuMuGoodMuons+ALCARECOTkAlHLTTracksZMuMuRelCombIsoMuons+ALCARECOTkAlHLTTracksZMuMu)


-- dummy change --
