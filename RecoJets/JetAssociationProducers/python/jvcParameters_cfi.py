import FWCore.ParameterSet.Config as cms

# $Id: ic5JetTracksAssociatorAtVertex_cfi.py,v 1.2 2008/04/21 03:27:42 rpw Exp $

jvcParameters = cms.PSet(
	primaryVertices = cms.InputTag("offlinePrimaryVertices"),
	cut = cms.double(3.0),
	temperature = cms.double(1.5)
)
