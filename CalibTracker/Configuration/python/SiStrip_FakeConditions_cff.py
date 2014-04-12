import FWCore.ParameterSet.Config as cms

#
# Strip calib
#
from CalibTracker.Configuration.SiStripCabling.SiStripCabling_Fake_cff import *
from CalibTracker.Configuration.SiStripGain.SiStripGain_Fake_cff import *
from CalibTracker.Configuration.SiStripLorentzAngle.SiStripLorentzAngle_Fake_cff import *
#include "CalibTracker/Configuration/data/SiStripNoise/SiStripNoise_Fake_APVModeDec.cff"
from CalibTracker.Configuration.SiStripNoise.SiStripNoise_Fake_APVModePeak_cff import *
from CalibTracker.Configuration.SiStripPedestals.SiStripPedestals_Fake_cff import *
from CalibTracker.Configuration.SiStripQuality.SiStripQuality_Fake_cff import *
from CalibTracker.Configuration.SiStripThreshold.SiStripThreshold_Fake_cff import *

