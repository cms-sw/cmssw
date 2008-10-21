import FWCore.ParameterSet.Config as cms

# File: MetMuonCorrections.cff
# Author: K. Terashi
# Date: 08.31.2007
#
# Met corrections for global muons
from Geometry.CaloEventSetup.CaloGeometry_cfi import *
from Geometry.CaloEventSetup.CaloTopology_cfi import *
from Geometry.CommonDetUnit.bareGlobalTrackingGeometry_cfi import *
from TrackingTools.TrackAssociator.default_cfi import *
corMetMuons = cms.EDFilter("MuonMET",
    TrackAssociatorParameterBlock,
    muonEtaRange = cms.double(2.5),
    inputMuonsLabel = cms.InputTag("muons"),
    muonPtMin = cms.double(10.0),
    muonDPtMax = cms.double(0.5),
    muonTrackD0Max = cms.double(999.0),
    inputUncorMetLabel = cms.InputTag("met"),
    muonTrackDzMax = cms.double(999.0),
    muonNHitsMin = cms.int32(5),
    muonChiSqMax = cms.double(1000.0),
    muonDepositCor = cms.bool(True),
    metType = cms.string('CaloMET')
)


