import FWCore.ParameterSet.Config as cms

# $Id: jvcParameters_cfi.py,v 1.1 2009/03/26 10:35:00 saout Exp $

jvcParameters = cms.PSet(
	primaryVertices = cms.InputTag("offlinePrimaryVertices"),
	cut = cms.double(3.0),
	temperature = cms.double(1.5)
)
