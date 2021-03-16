import FWCore.ParameterSet.Config as cms

# define the HLT base path
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOPPSCalTrackBasedSelHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True,
    HLTPaths = ['HLT_ZeroBias_v*'],
    #eventSetupPathKey = 'SiStripCalZeroBias', #FIXME find a proper base key
    throw = False
)

# perform basic PPS reconstruction
from EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff import *
from RecoPPS.Configuration.recoCTPPS_cff import *

# select events passing the filter on pixel tracks
from CalibPPS.TimingCalibration.ppsTrackFilter_cfi import ppsTrackFilter

seqALCARECOPPSCalTrackBasedSel = cms.Sequence(
    ctppsRawToDigi *
    recoCTPPS *
    ALCARECOPPSCalTrackBasedSelHLT *
    ppsTrackFilter
)

