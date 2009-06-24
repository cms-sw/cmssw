import FWCore.ParameterSet.Config as cms

# This HLTFilter will accept only events with HCAL zero suppression enabled.
hltHcalZSFilter = cms.EDFilter("HLTHcalZSFilter")

