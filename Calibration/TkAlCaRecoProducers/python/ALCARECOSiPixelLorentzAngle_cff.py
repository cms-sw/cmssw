import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
# AlCaReco for track based calibration using min. bias events
ALCARECOSiPixelLorentzAngleHLTFilter = copy.deepcopy(hltHighLevel)
seqALCARECOSiPixelLorentzAngle = cms.Sequence(ALCARECOSiPixelLorentzAngleHLTFilter)
ALCARECOSiPixelLorentzAngleHLTFilter.andOr = True ## choose logical OR between Triggerbits

ALCARECOSiPixelLorentzAngleHLTFilter.HLTPaths = ['HLT1MuonIso', 'HLT2MuonNonIso', 'HLT2MuonJPsi', 'HLT2MuonUpsilon', 'HLT2MuonZ', 
    'HLT2MuonSameSign']

