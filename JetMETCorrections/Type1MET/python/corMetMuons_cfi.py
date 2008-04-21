import FWCore.ParameterSet.Config as cms

# File: MetMuonCorrections.cff
# Author: K. Terashi
# Date: 08.31.2007
#
# Met corrections for global muons
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
from Geometry.CaloEventSetup.CaloGeometry_cfi import *
from Geometry.CaloEventSetup.CaloTopology_cfi import *
from Geometry.DTGeometry.dtGeometry_cfi import *
from Geometry.RPCGeometry.rpcGeometry_cfi import *
from Geometry.CSCGeometry.cscGeometry_cfi import *
from Geometry.CommonDetUnit.bareGlobalTrackingGeometry_cfi import *
from Geometry.MuonNumbering.muonNumberingInitialization_cfi import *
from TrackingTools.TrackAssociator.default_cfi import *
corMetMuons = cms.EDFilter("MuonMET",
    TrackAssociatorParameterBlock,
    muonEtaRange = cms.double(2.5),
    inputMuonsLabel = cms.string('muons'),
    muonPtMin = cms.double(10.0),
    muonDPtMax = cms.double(0.5),
    muonTrackD0Max = cms.double(999.0),
    inputUncorMetLabel = cms.string('met'),
    muonTrackDzMax = cms.double(999.0),
    muonNHitsMin = cms.int32(5),
    muonChiSqMax = cms.double(1000.0),
    muonDepositCor = cms.bool(True),
    metType = cms.string('CaloMET')
)


