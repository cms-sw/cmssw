# AlCaReco for muon alignment using global cosmic ray tracks
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
    AndOr        = cms.bool(False),
    DebugOn      = cms.untracked.bool(False)
)

import Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi

ALCARECOMuAlGlobalCosmics = Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi.AlignmentMuonSelector.clone(
    src = cms.InputTag("muons"),
    filter = True, # not strictly necessary, but provided for symmetry with MuAlStandAloneCosmics
    nHitMinGB = 1,
    ptMin = 10.0,
    etaMin = -100.0,
    etaMax =  100.0
    )



#________________________________Track selection____________________________________
# AlCaReco for track based alignment using Cosmic muons reconstructed by Combinatorial Track Finder

import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOMuAlCosmicsCTF = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone(
    src = 'ctfWithMaterialTracksP5',
    filter = True,
    applyBasicCuts = True,
    ptMin = 0., ##10
    ptMax = 99999.,
    pMin = 4., ##10
    pMax = 99999.,
    etaMin = -99., 
    etaMax = 99., 

    nHitMin = 7,
    nHitMin2D = 2,
    chi2nMax = 999999.,

    applyMultiplicityFilter = False,
    applyNHighestPt = True, ## select only highest pT track
    nHighestPt = 1
    )

# AlCaReco for track based alignment using Cosmic muons reconstructed by Cosmic Track Finder
# (same cuts)
ALCARECOMuAlCosmicsCosmicTF = ALCARECOMuAlCosmicsCTF.clone(
    src = 'cosmictrackfinderP5' ## different for CTF
    )

# AlCaReco for track based alignment using Cosmic muons reconstructed by Regional Cosmic Tracking
# (same cuts)
ALCARECOMuAlCosmicsRegional = ALCARECOMuAlCosmicsCTF.clone(
    src = 'regionalCosmicTracks'
    )

#________________________________Sequences____________________________________  

seqALCARECOMuAlGlobalCosmics = cms.Sequence(ALCARECOMuAlGlobalCosmicsHLT + ALCARECOMuAlGlobalCosmicsDCSFilter + ALCARECOMuAlGlobalCosmics)

seqALCARECOMuAlCosmicsCTF = cms.Sequence(ALCARECOMuAlCosmicsCTF)
seqALCARECOMuAlCosmicsCosmicTF = cms.Sequence(ALCARECOMuAlCosmicsCosmicTF)
seqALCARECOMuAlCosmicsRegional = cms.Sequence(ALCARECOMuAlCosmicsRegional)
