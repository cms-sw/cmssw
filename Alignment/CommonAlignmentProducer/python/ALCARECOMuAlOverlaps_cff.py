import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOMuAlOverlapsHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    eventSetupPathsKey = 'MuAlOverlaps',
    throw = False # tolerate triggers not available
)

# DCS partitions
# "EBp","EBm","EEp","EEm","HBHEa","HBHEb","HBHEc","HF","HO","RPC"
# "DT0","DTp","DTm","CSCp","CSCm","CASTOR","TIBTID","TOB","TECp","TECm"
# "BPIX","FPIX","ESp","ESm"
import DPGAnalysis.Skims.skim_detstatus_cfi
ALCARECOMuAlOverlapsDCSFilter = DPGAnalysis.Skims.skim_detstatus_cfi.dcsstatus.clone(
    DetectorType = cms.vstring('CSCp','CSCm'),
    ApplyFilter  = cms.bool(True),
    AndOr        = cms.bool(False), # False = at least one detector from DetectorType map above is ON
    DebugOn      = cms.untracked.bool(False),
)

#________________________________Event selection____________________________________
ALCARECOMuAlOverlaps = cms.EDFilter("AlignmentCSCOverlapSelectorModule",
    filter            = cms.bool(True),
    src               = cms.InputTag("ALCARECOMuAlOverlapsMuonSelector","StandAlone"),
    minHitsPerChamber = cms.uint32(1),
    station           = cms.int32(0) ## all stations: the algorithm can handle multiple stations now
)

#________________________________Muon selection____________________________________
# AlCaReco selected muons for track based muon alignment
import Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi
ALCARECOMuAlOverlapsMuonSelector = Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi.AlignmentMuonSelector.clone(
    ptMin = cms.double(3.0),
    etaMin = cms.double(-2.6),
    etaMax = cms.double(2.6),
)

#________________________________Track selection____________________________________
# AlCaReco selected general tracks for track based muon alignment
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOMuAlOverlapsGeneralTracks = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone(
    src = cms.InputTag("generalTracks"),
    filter = cms.bool(True),
    ptMin = cms.double(2.0),
    etaMin = cms.double(-2.6),
    etaMax = cms.double(2.6),
    nHitMin = cms.double(7),
)

seqALCARECOMuAlOverlaps = cms.Sequence(ALCARECOMuAlOverlapsHLT+ALCARECOMuAlOverlapsDCSFilter+ALCARECOMuAlOverlapsMuonSelector*ALCARECOMuAlOverlaps)

seqALCARECOMuAlOverlapsGeneralTracks = cms.Sequence(ALCARECOMuAlOverlapsGeneralTracks)
