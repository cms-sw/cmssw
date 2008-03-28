import FWCore.ParameterSet.Config as cms

# File: MetMuonCorrections.cff
# Author: K. Terashi
# Date: 08.31.2007
#
# Met corrections for global muons
from Geometry.CMSCommonData.cmsSimIdealGeometryXML_cfi import *
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
from Geometry.CaloEventSetup.CaloGeometry_cfi import *
from Geometry.CaloEventSetup.CaloTopology_cfi import *
from Geometry.DTGeometry.dtGeometry_cfi import *
from Geometry.RPCGeometry.rpcGeometry_cfi import *
from Geometry.CSCGeometry.cscGeometry_cfi import *
from Geometry.CommonDetUnit.bareGlobalTrackingGeometry_cfi import *
from Geometry.MuonNumbering.muonNumberingInitialization_cfi import *
corMetGlobalMuons = cms.EDFilter("MuonMET",
    muonTrackDzMax = cms.double(999.0),
    muonEtaRange = cms.double(2.5),
    inputMuonsLabel = cms.string('muons'),
    muonPtMin = cms.double(10.0),
    muonDPtMax = cms.double(0.5),
    muonTrackD0Max = cms.double(999.0),
    inputUncorMetLabel = cms.string('met'),
    TrackAssociatorParameters = cms.PSet(
        muonMaxDistanceSigmaX = cms.double(0.0),
        muonMaxDistanceSigmaY = cms.double(0.0),
        CSCSegmentCollectionLabel = cms.InputTag("cscSegments"),
        dRHcal = cms.double(9999.0),
        dREcal = cms.double(9999.0),
        CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
        useEcal = cms.bool(True),
        dREcalPreselection = cms.double(0.05),
        HORecHitCollectionLabel = cms.InputTag("horeco"),
        dRMuon = cms.double(9999.0),
        crossedEnergyType = cms.string('SinglePointAlongTrajectory'),
        muonMaxDistanceX = cms.double(5.0),
        muonMaxDistanceY = cms.double(5.0),
        useHO = cms.bool(True),
        accountForTrajectoryChangeCalo = cms.bool(False),
        DTRecSegment4DCollectionLabel = cms.InputTag("dt4DSegments"),
        EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
        dRHcalPreselection = cms.double(0.2),
        useMuon = cms.bool(True),
        useCalo = cms.bool(False),
        EBRecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
        dRMuonPreselection = cms.double(0.2),
        truthMatch = cms.bool(False),
        HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
        useHcal = cms.bool(True)
    ),
    muonNHitsMin = cms.int32(5),
    muonChiSqMax = cms.double(1000.0),
    muonDepositCor = cms.bool(True),
    metType = cms.string('CaloMET')
)

MetMuonCorrections = cms.Sequence(corMetGlobalMuons)
corMetGlobalMuons.TrackAssociatorParameters.useCalo = True
corMetGlobalMuons.TrackAssociatorParameters.truthMatch = False

