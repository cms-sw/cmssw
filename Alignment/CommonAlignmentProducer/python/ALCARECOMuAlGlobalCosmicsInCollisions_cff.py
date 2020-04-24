# AlCaReco for muon alignment using global cosmic ray tracks taken during collisions
import FWCore.ParameterSet.Config as cms

# HLT
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOMuAlGlobalCosmicsInCollisionsHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    eventSetupPathsKey = 'MuAlGlobalCosmicsInCollisions',
    throw = False # tolerate triggers not available
    )

# DCS partitions
# "EBp","EBm","EEp","EEm","HBHEa","HBHEb","HBHEc","HF","HO","RPC"
# "DT0","DTp","DTm","CSCp","CSCm","CASTOR","TIBTID","TOB","TECp","TECm"
# "BPIX","FPIX","ESp","ESm"
import DPGAnalysis.Skims.skim_detstatus_cfi
ALCARECOMuAlGlobalCosmicsInCollisionsDCSFilter = DPGAnalysis.Skims.skim_detstatus_cfi.dcsstatus.clone(
    DetectorType = cms.vstring('DT0','DTp','DTm'),
    ApplyFilter  = cms.bool(True),
    AndOr        = cms.bool(False),
    DebugOn      = cms.untracked.bool(False)
)

import Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi

ALCARECOMuAlGlobalCosmicsInCollisions = Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi.AlignmentMuonSelector.clone(
    src = cms.InputTag("muonsFromCosmics"),
    filter = True, # not strictly necessary, but provided for symmetry with MuAlStandAloneCosmics
    nHitMinGB = 1,
    ptMin = 10.0,
    etaMin = -100.0,
    etaMax =  100.0
    )

seqALCARECOMuAlGlobalCosmicsInCollisions = cms.Sequence(ALCARECOMuAlGlobalCosmicsInCollisionsHLT + ALCARECOMuAlGlobalCosmicsInCollisionsDCSFilter + ALCARECOMuAlGlobalCosmicsInCollisions)
