import FWCore.ParameterSet.Config as cms
import sys

sys.stderr.write("WARNING:  L1Trigger/L1TCommon/python/caloStage1LegacyFormatDigis_cfi.py has been deprecated...\n")
sys.stderr.write("WARNING:  please use L1Trigger/L1TCalorimeter/python/caloStage1LegacyFormatDigis_cfi.py\n")

from L1Trigger.L1TCalorimeter.caloStage1LegacyFormatDigis_cfi import caloStage1LegacyFormatDigis


