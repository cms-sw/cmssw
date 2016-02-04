import FWCore.ParameterSet.Config as cms
from L1Trigger.Configuration.L1Config_cff import *
from CalibCalorimetry.CaloTPG.CaloTPGTranscoder_cfi import *
from CalibCalorimetry.EcalTPGTools.ecalTPGScale_cff import *
from L1TriggerConfig.RCTConfigProducers.L1RCTConfig_cff import *

CaloTPGTranscoder.hcalLUT2 = 'TPGcalcDecompress2Identity.txt'
EcalTrigPrimESProducer.DatabaseFile = 'TPG_startup.txt.gz'
l1CaloScales.L1CaloEmEtScaleLSB = 1.

RCTConfigProducers.eGammaLSB = 1.
RCTConfigProducers.jetMETLSB = 1.
RCTConfigProducers.eMinForFGCut = 999. ## FG cut not used, this serves

RCTConfigProducers.eMaxForFGCut = -999. ## to disable it.

RCTConfigProducers.hOeCut = 999. ## H/E not used

RCTConfigProducers.eMinForHoECut = 999. ## H/E cut not used,

RCTConfigProducers.eMaxForHoECut = -999. ## disabled here

RCTConfigProducers.hMinForHoECut = 999. ##

RCTConfigProducers.eActivityCut = 2. ## Activity bits for tau calc

RCTConfigProducers.hActivityCut = 2. ## 

RCTConfigProducers.eicIsolationThreshold = 0 ## Force non-isolation

# The following vectors determine paths used
# Eeg
RCTConfigProducers.eGammaECalScaleFactors = [0., 0., 0., 0., 0., 
    0., 0., 0., 0., 0., 
    0., 0., 0., 0., 0., 
    0., 0., 0., 0., 0., 
    0., 0., 0., 0., 0., 
    0., 0., 0.]
# Heg
RCTConfigProducers.eGammaHCalScaleFactors = [1., 1., 1., 1., 1., 
    1., 1., 1., 1., 1., 
    1., 1., 1., 1., 1., 
    1., 1., 1., 1., 1., 
    1., 1., 1., 1., 1., 
    1., 1., 1.]
# ESums
RCTConfigProducers.jetMETECalScaleFactors = [0., 0., 0., 0., 0., 
    0., 0., 0., 0., 0., 
    0., 0., 0., 0., 0., 
    0., 0., 0., 0., 0., 
    0., 0., 0., 0., 0., 
    0., 0., 0.]
# HSums
RCTConfigProducers.jetMETHCalScaleFactors = [1., 1., 1., 1., 1., 
    1., 1., 1., 1., 1., 
    1., 1., 1., 1., 1., 
    1., 1., 1., 1., 1., 
    1., 1., 1., 1., 1., 
    1., 1., 1.]

