import FWCore.ParameterSet.Config as cms

# HLT
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOMuAlGlobalCosmicsHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    eventSetupPathsKey = 'MuAlGlobalCosmics',
    throw = False # tolerate triggers not available
)

# DCS partitions
# "EBp","EBm","EEp","EEm","HBHEa","HBHEb","HBHEc","HF","HO","RPC"
# "DT0","DTp","DTm","CSCp","CSCm","CASTOR","TIBTID","TOB","TECp","TECm"
# "BPIX","FPIX","ESp","ESm"
import DPGAnalysis.Skims.skim_detstatus_cfi
ALCARECOMuAlGlobalCosmicsDCSFilter = DPGAnalysis.Skims.skim_detstatus_cfi.dcsstatus.clone(
    DetectorType = cms.vstring('DT0','DTp','DTm',"CSCp","CSCm"),
    ApplyFilter  = cms.bool(True),
    AndOr        = cms.bool(False), # False = at least one detector from DetectorType map above is ON
    DebugOn      = cms.untracked.bool(False)
)

#________________________________Muon selection____________________________________
# AlCaReco selected muons for track based muon alignment
import Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi
ALCARECOMuAlGlobalCosmics = Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi.AlignmentMuonSelector.clone(
    src       = cms.InputTag("muons"),
    filter    = cms.bool(True), # not strictly necessary, but provided for symmetry with MuAlStandAloneCosmics
    ptMin     = cms.double(10.0),
    etaMin    = cms.double(-100.0),
    etaMax    = cms.double(100.0),
)

# AlCaReco selected Combinatorial Track Finder tracks for track based muon alignment
# (same cuts)
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOMuAlGlobalCosmicsCombinatorialTF = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone(
    src = 'ctfWithMaterialTracksP5',
    filter = cms.bool(True),
    ptMin = cms.double(8.0),
    etaMin = cms.double(-100.0),
    etaMax = cms.double(100.0),
    nHitMin = cms.double(7),
    applyNHighestPt = cms.bool(True), ## select only 3 highest pT tracks
    nHighestPt = cms.int32(3),
    )

# AlCaReco selected Cosmic Track Finder tracks for track based muon alignment
# (same cuts)
ALCARECOMuAlGlobalCosmicsCosmicTF = ALCARECOMuAlGlobalCosmicsCombinatorialTF.clone(
    src = 'cosmictrackfinderP5'
)

# AlCaReco selected Regional Cosmic Tracking tracks for track based muon alignment
# (same cuts)
ALCARECOMuAlGlobalCosmicsRegionalTF = ALCARECOMuAlGlobalCosmicsCombinatorialTF.clone(
    src = 'regionalCosmicTracks'
)

#________________________________Sequences____________________________________  
seqALCARECOMuAlGlobalCosmics = cms.Sequence(ALCARECOMuAlGlobalCosmicsHLT + ALCARECOMuAlGlobalCosmicsDCSFilter + ALCARECOMuAlGlobalCosmics)
seqALCARECOMuAlGlobalCosmicsCombinatorialTF = cms.Sequence(ALCARECOMuAlGlobalCosmicsCombinatorialTF)
seqALCARECOMuAlGlobalCosmicsCosmicTF        = cms.Sequence(ALCARECOMuAlGlobalCosmicsCosmicTF)
seqALCARECOMuAlGlobalCosmicsRegionalTF      = cms.Sequence(ALCARECOMuAlGlobalCosmicsRegionalTF)
