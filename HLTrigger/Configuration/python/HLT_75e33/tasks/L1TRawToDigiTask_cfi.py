import FWCore.ParameterSet.Config as cms

from ..tasks.L1TRawToDigi_Stage1_cfi import *
from ..tasks.L1TRawToDigi_Stage2_cfi import *

L1TRawToDigiTask = cms.Task(L1TRawToDigi_Stage1, L1TRawToDigi_Stage2)
