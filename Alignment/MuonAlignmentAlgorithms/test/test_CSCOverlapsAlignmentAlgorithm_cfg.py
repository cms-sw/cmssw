import FWCore.ParameterSet.Config as cms

process = cms.Process("CSCOverlapsAlignmentAlgorithm")
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))
process.CSCOverlapsTrackPreparation = cms.EDProducer("CSCOverlapsTrackPreparation", src = cms.InputTag("ALCARECOMuAlBeamHaloOverlaps"))
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")
process.Path = cms.Path(process.offlineBeamSpot+process.CSCOverlapsTrackPreparation)

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.CommonDetUnit.bareGlobalTrackingGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("Alignment.CommonAlignmentProducer.AlignmentProducer_cff")
process.looper.doTracker = cms.untracked.bool(False)
process.looper.doMuon = cms.untracked.bool(True)
process.looper.tjTkAssociationMapTag = cms.InputTag("CSCOverlapsTrackPreparation")
process.looper.algoConfig = cms.PSet(
  algoName = cms.string("CSCOverlapsAlignmentAlgorithm"),
  mode = cms.string("phipos"),
  maxHitErr = cms.double(0.2),
  minHitsPerChamber = cms.int32(6),
  maxRotYDiff = cms.double(0.030),
  maxRPhiDiff = cms.double(1.5),
  maxRedChi2 = cms.double(10.),
  minTracksPerAlignable = cms.int32(10),
  useHitWeightsInTrackFit = cms.bool(True),
  useFitWeightsInMean = cms.bool(False),
  makeHistograms = cms.bool(True),
  )

import Alignment.MuonAlignmentAlgorithms.MuonStationSelectors_cff
params = dict(Alignment.MuonAlignmentAlgorithms.MuonStationSelectors_cff.MuonStationSelectors)
params.update({"alignParams": cms.vstring(
  "MuonCSCChambers,110001,meplus21",
  "MuonCSCChambers,110001,meminus21",
  "MuonCSCChambers,110001,meminus31",
  )})
process.looper.ParameterBuilder.Selector = cms.PSet(**params)

process.load("CalibMuon.Configuration.Muon_FakeAlignment_cff")

process.TFileService = cms.Service("TFileService", fileName = cms.string("layerplots.root"))
