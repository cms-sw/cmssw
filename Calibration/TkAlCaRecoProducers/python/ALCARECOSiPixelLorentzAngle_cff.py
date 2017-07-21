import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
# AlCaReco for track based calibration using min. bias events
ALCARECOSiPixelLorentzAngleHLTFilter = copy.deepcopy(hltHighLevel)
seqALCARECOSiPixelLorentzAngle = cms.Sequence(ALCARECOSiPixelLorentzAngleHLTFilter)
ALCARECOSiPixelLorentzAngleHLTFilter.andOr = True ## choose logical OR between Triggerbits
ALCARECOSiPixelLorentzAngleHLTFilter.throw = False ## dont throw on unknown path names
ALCARECOSiPixelLorentzAngleHLTFilter.eventSetupPathsKey = 'SiPixelLorentzAngle'

# ALCARECOSiPixelLorentzAngleHLTFilter.HLTPaths = ['HLT_IsoMu11', 'HLT_DoubleMu3', 'HLT_DoubleMu3_JPsi', 'HLT_DoubleMu3_Upsilon', 'HLT_DoubleMu7_Z',
#     'HLT_DoubleMu3_SameSign']
# FIXME: Put this into the trigger bits payload and remove the commented lines completely
