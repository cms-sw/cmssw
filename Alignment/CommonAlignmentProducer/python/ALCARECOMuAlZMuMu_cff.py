# AlCaReco for muon based alignment using ZMuMu events
import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOMuAlZMuMuHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    eventSetupPathsKey = 'MuAlZMuMu',
    throw = False # tolerate triggers not available
)

# DCS partitions
# "EBp","EBm","EEp","EEm","HBHEa","HBHEb","HBHEc","HF","HO","RPC"
# "DT0","DTp","DTm","CSCp","CSCm","CASTOR","TIBTID","TOB","TECp","TECm"
# "BPIX","FPIX","ESp","ESm"
import DPGAnalysis.Skims.skim_detstatus_cfi
ALCARECOMuAlZMuMuDCSFilter = DPGAnalysis.Skims.skim_detstatus_cfi.dcsstatus.clone(
    DetectorType = cms.vstring('TIBTID','TOB','DT0','DTp','DTm','CSCp','CSCm'),
    ApplyFilter  = cms.bool(True),
    AndOr        = cms.bool(True),
    DebugOn      = cms.untracked.bool(False)
)

#________________________________Muon selection____________________________________
# AlCaReco selected muons for track based muon alignment
import Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi
ALCARECOMuAlZMuMu = Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi.AlignmentMuonSelector.clone(
    src                 = cms.InputTag("muons"),
    ptMin               = cms.double(5.0),
    etaMin              = cms.double(-2.6),
    etaMax              = cms.double(2.6),
    applyMassPairFilter = cms.bool(True),
    minMassPair         = cms.double(81.0),  # 91 GeV - 10 GeV
    maxMassPair         = cms.double(101.0), # 91 GeV + 10 GeV
)

#________________________________Track selection____________________________________
# AlCaReco selected general tracks for track based muon alignment
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOMuAlZMuMuGeneralTracks = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone(
    src     = cms.InputTag("generalTracks"),
    filter  = cms.bool(True),
    ptMin   = cms.double(5.0),
    etaMin  = cms.double(-2.6),
    etaMax  = cms.double(2.6),
    nHitMin = cms.double(7),
)

#________________________________Sequences____________________________________
seqALCARECOMuAlZMuMu = cms.Sequence(ALCARECOMuAlZMuMuHLT+ALCARECOMuAlZMuMuDCSFilter+ALCARECOMuAlZMuMu)

seqALCARECOMuAlZMuMuGeneralTracks = cms.Sequence(ALCARECOMuAlZMuMuGeneralTracks)
