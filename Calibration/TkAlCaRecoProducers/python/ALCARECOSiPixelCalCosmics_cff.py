import FWCore.ParameterSet.Config as cms
import copy

from ALCARECOSiPixelCalSingleMuon_cff import ALCARECOSiPixelCalSingleMuonHLTFilter 
ALCARECOSiPixelCalCosmicsHLTFilter = ALCARECOSiPixelCalSingleMuonHLTFilter.clone(
    HLTPaths = ["HLT_*"],
    eventSetupPathsKey = ''
)

from ALCARECOSiPixelCalSingleMuon_cff import ALCARECOSiPixelCalSingleMuon 
ALCARECOSiPixelCalCosmics = ALCARECOSiPixelCalSingleMuon.clone(
    ptMin = 0.0, #GeV
    src = 'ctfWithMaterialTracksP5'
)

# Sequence #
seqALCARECOSiPixelCalCosmics = cms.Sequence(ALCARECOSiPixelCalCosmicsHLTFilter*ALCARECOSiPixelCalCosmics)

