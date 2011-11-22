import FWCore.ParameterSet.Config as cms

# producer for alcaisotrk (HCAL isolated tracks)
from TrackingTools.TrackAssociator.default_cfi import *
from TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff import *

TrackAssociatorParameterBlock.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("IsoProd","IsoTrackEcalRecHitCollection")
TrackAssociatorParameterBlock.TrackAssociatorParameters.EBRecHitCollectionLabel = cms.InputTag("IsoProd","IsoTrackEcalRecHitCollection")
TrackAssociatorParameterBlock.TrackAssociatorParameters.HBHERecHitCollectionLabel = cms.InputTag("IsoProd","IsoTrackHBHERecHitCollection")
TrackAssociatorParameterBlock.TrackAssociatorParameters.HORecHitCollectionLabel = cms.InputTag("IsoProd","IsoTrackHORecHitCollection")

isoAnalyzer = cms.EDAnalyzer("HcalIsoTrkAnalyzer",
	TrackAssociatorParameterBlock,
        hbheInput = cms.InputTag("IsoProd:IsoTrackHBHERecHitCollection"),
        hoInput = cms.InputTag("IsoProd:IsoTrackHORecHitCollection"),
        eInput = cms.InputTag("IsoProd:IsoTrackEcalRecHitCollection"),
        HcalIsolTrackInput = cms.InputTag("IsoProd:HcalIsolatedTrackCollection"),
        trackInput = cms.InputTag("IsoProd:IsoTrackTracksCollection"),
        associationConeSize = cms.double(0.5),
        outputFileName = cms.string("test_IsoAn.root"),
        AxB = cms.string("Cone"),
        calibrationConeSize = cms.double(40.),

        noOfIterations = cms.int32(10),
        eventWeight = cms.double(2.0),
        energyMinIso = cms.double(10.0),
        energyMaxIso = cms.double(1000.0),
        energyECALmip = cms.double(1.0),
	maxPNear = cms.double(2.0),
	MinNTrackHitsBarrel = cms.int32(13),
	MinNTECHitsEndcap = cms.int32(11)
)


