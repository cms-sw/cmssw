import FWCore.ParameterSet.Config as cms

from EventFilter.L1TRawToDigi.bmtfDigis_cfi import *
from L1Trigger.L1TMuonBarrel.simKBmtfStubs_cfi import *
simKBmtfStubs.srcPhi = cms.InputTag("bmtfDigis")
simKBmtfStubs.srcTheta = cms.InputTag("bmtfDigis")

from L1Trigger.L1TMuonBarrel.simKBmtfDigis_cfi import * #Kalman
from L1Trigger.L1TMuonBarrel.simBmtfDigis_cfi import * #BMTF
simBmtfDigis.DTDigi_Source = cms.InputTag("bmtfDigis")
simBmtfDigis.DTDigi_Theta_Source = cms.InputTag("bmtfDigis")

kbmtf = cms.Task(bmtfDigis,simBmtfDigis,simKBmtfStubs,simKBmtfDigis)
