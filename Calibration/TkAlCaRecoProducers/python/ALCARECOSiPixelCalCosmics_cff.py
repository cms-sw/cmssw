import FWCore.ParameterSet.Config as cms
import copy

# DCS partitions
# "EBp","EBm","EEp","EEm","HBHEa","HBHEb","HBHEc","HF","HO","RPC"
# "DT0","DTp","DTm","CSCp","CSCm","CASTOR","TIBTID","TOB","TECp","TECm"
# "BPIX","FPIX","ESp","ESm"
import DPGAnalysis.Skims.skim_detstatus_cfi
ALCARECOSiPixelCalCosmicsDCSFilter = DPGAnalysis.Skims.skim_detstatus_cfi.dcsstatus.clone(
    DetectorType = cms.vstring('BPIX','FPIX'),
    ApplyFilter  = cms.bool(True),
    AndOr        = cms.bool(True),
    DebugOn      = cms.untracked.bool(False)
)

from ALCARECOSiPixelCalSingleMuon_cff import ALCARECOSiPixelCalSingleMuonHLTFilter 
ALCARECOSiPixelCalCosmicsHLTFilter = ALCARECOSiPixelCalSingleMuonHLTFilter.clone(
    HLTPaths = ["HLT_*"],
    eventSetupPathsKey = ''
)

from ALCARECOSiPixelCalSingleMuon_cff import ALCARECOSiPixelCalSingleMuon
ALCARECOSiPixelCalCosmics = ALCARECOSiPixelCalSingleMuon.clone(
    filter = True,
    applyBasicCuts = True,
    ptMin = 3.,
    ptMax = 99999.,
    pMin = 0.,
    pMax = 99999.,
    etaMin = -99., ##-2.4 keep also what is going through...
    etaMax = 99., ## 2.4 ...both TEC with flat slope
    chi2nMax = 3.,
    applyMultiplicityFilter = False,
    applyNHighestPt = False, ## select only highest pT track
    src = 'ctfWithMaterialTracksP5',
)
ALCARECOSiPixelCalCosmics.minHitsPerSubDet.inPIXEL = 1

# Sequence #
seqALCARECOSiPixelCalCosmics = cms.Sequence(ALCARECOSiPixelCalCosmicsDCSFilter+ALCARECOSiPixelCalCosmicsHLTFilter*ALCARECOSiPixelCalCosmics)
