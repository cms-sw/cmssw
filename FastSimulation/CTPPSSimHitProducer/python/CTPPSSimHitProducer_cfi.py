import FWCore.ParameterSet.Config as cms

CTPPSSimHits = cms.EDProducer('CTPPSSimHitProducer',
    MCEvent = cms.untracked.InputTag("LHCTransport"),
    Z_Tracker1 = cms.double(203.827),# first tracker z position in m
    Z_Tracker2 = cms.double(212.550),# second tracker z position in m 	
    Z_Timing =  cms.double(215.700) # timing detector z position in m


)
