import FWCore.ParameterSet.Config as cms

# This modifier sets the LST (Phase-2 line segment tracking) used for track seeding
# Needs to be used on top of the trackingLST modifier
seedingLST = cms.Modifier()
