import FWCore.ParameterSet.Config as cms

# define the HLT base path
from HLTrigger.HLTfilters.hltHighLevel_cfi import hltHighLevel as _hlt
ALCARECOPPSCalTrackBasedSelHLT = _hlt.clone(
    andOr = True,
    HLTPaths = ['HLT_ZeroBias_v*'],
    #eventSetupPathKey = 'SiStripCalZeroBias', # in case we have a proper base key
    throw = False
)

# perform basic PPS reconstruction
from EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff import *
from RecoPPS.Configuration.recoCTPPS_cff import *

# select events passing the filter on pixel tracks
from HLTrigger.special.hltPPSPerPotTrackFilter_cfi import hltPPSPerPotTrackFilter as _filter
hltPPSPerPotTrackFilter = _filter.clone(
    pixelFilter = cms.VPSet(
        cms.PSet( # sector 45, near pot
            detid = cms.uint32(2022703104),
            minTracks = cms.int32(1),
            maxTracks = cms.int32(6),
        ),
        cms.PSet( # sector 45, far pot
            detid = cms.uint32(2023227392),
            minTracks = cms.int32(1),
            maxTracks = cms.int32(6),
        ),
        cms.PSet( # sector 56, near pot
            detid = cms.uint32(2039480320),
            minTracks = cms.int32(1),
            maxTracks = cms.int32(6),
        ),
        cms.PSet( # sector 56, far pot
            detid = cms.uint32(2040004608),
            minTracks = cms.int32(1),
            maxTracks = cms.int32(6),
        ),
    )
)

seqALCARECOPPSCalTrackBasedSel = cms.Sequence(
    ctppsRawToDigi *
    recoCTPPS *
    ALCARECOPPSCalTrackBasedSelHLT *
    hltPPSPerPotTrackFilter
)
