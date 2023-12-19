import FWCore.ParameterSet.Config as cms
import os
from L1Trigger.VertexFinder.l1tVertexProducer_cfi import l1tVertexProducer
CMSSW_BASE = os.getenv('CMSSW_BASE')

l1tTrackVertexAssociationProducer = cms.EDProducer('L1TrackVertexAssociationProducer',
  l1TracksInputTag = cms.InputTag("l1tGTTInputProducer","Level1TTTracksConverted"),
  l1SelectedTracksInputTag = cms.InputTag("l1tTrackSelectionProducer", "Level1TTTracksSelected"),
  l1SelectedTracksEmulationInputTag = cms.InputTag("l1tTrackSelectionProducer", "Level1TTTracksSelectedEmulation"),
  # If no vertex collection is provided, then the DeltaZ cuts will not be run
  l1VerticesInputTag = cms.InputTag("l1tVertexFinder", "L1Vertices"),
  l1VerticesEmulationInputTag = cms.InputTag("l1tVertexFinderEmulator", "L1VerticesEmulation"),
  outputCollectionName = cms.string("Level1TTTracksSelectedAssociated"),
  cutSet = cms.PSet(
                    #deltaZMaxEtaBounds = cms.vdouble(0.0, absEtaMax.value), # these values define the bin boundaries in |eta|
                    #deltaZMax = cms.vdouble(0.5), # delta z must be less than these values, there will be one less value here than in deltaZMaxEtaBounds, [cm]
                    deltaZMaxEtaBounds = cms.vdouble(0.0, 0.7, 1.0, 1.2, 1.6, 2.0, 2.4), # these values define the bin boundaries in |eta|
                    deltaZMax = cms.vdouble(0.37, 0.50, 0.60, 0.75, 1.00, 1.60), # delta z must be less than these values, there will be one less value here than in deltaZMaxEtaBounds, [cm]
                    ),
  useDisplacedTracksDeltaZOverride = cms.double(-1.0), # override the deltaZ cut value for displaced tracks
  processSimulatedTracks = cms.bool(True), # return selected tracks after cutting on the floating point values
  processEmulatedTracks = cms.bool(True), # return selected tracks after cutting on the bitwise emulated values
  fwNTrackSetsTVA = cms.uint32(94), # firmware limit on number of GTT converted tracks considered for primary vertex association
  debug = cms.int32(0), # Verbosity levels: 0, 1, 2, 3, 4
)

l1tTrackVertexNNAssociationProducer = l1tTrackVertexAssociationProducer.clone(
  processSimulatedTracks = cms.bool(True), # return selected tracks after cutting on the floating point values
  processEmulatedTracks = cms.bool(True), # return selected tracks after cutting on the bitwise emulated values
  useAssociationNetwork = cms.bool(True), #Enable Association Network
  associationThreshold = cms.double(0.1), #Association Network threshold for PV tracks
  associationGraph = cms.string(CMSSW_BASE+"/src/L1Trigger/L1TTrackMatch/data/Quantised_model_prune_iteration_9_associationModelgraph.pb"), #Location of Association Network model file
  associationNetworkZ0binning = l1tVertexProducer.VertexReconstruction.FH_HistogramParameters, #Z0 binning used for setting the input feature digitisation
  associationNetworkEtaBounds = cms.vdouble(0.0, 0.01984126984126984, 0.03968253968253968, 0.05952380952380952, 0.07936507936507936, 0.0992063492063492, 0.11904761904761904, 0.1388888888888889, 0.15873015873015872, 0.17857142857142855, 0.1984126984126984, 0.21825396825396826, 0.23809523809523808, 0.2579365079365079, 0.2777777777777778, 0.2976190476190476, 0.31746031746031744, 0.33730158730158727, 0.3571428571428571, 0.376984126984127, 0.3968253968253968, 0.41666666666666663, 0.4365079365079365, 0.45634920634920634, 0.47619047619047616, 0.496031746031746, 0.5158730158730158, 0.5357142857142857, 0.5555555555555556, 0.5753968253968254, 0.5952380952380952, 0.615079365079365, 0.6349206349206349, 0.6547619047619048, 0.6746031746031745, 0.6944444444444444, 0.7142857142857142, 0.7341269841269841, 0.753968253968254, 0.7738095238095237, 0.7936507936507936, 0.8134920634920635, 0.8333333333333333, 0.8531746031746031, 0.873015873015873, 0.8928571428571428, 0.9126984126984127, 0.9325396825396824, 0.9523809523809523, 0.9722222222222222, 0.992063492063492, 1.0119047619047619, 1.0317460317460316, 1.0515873015873016, 1.0714285714285714, 1.0912698412698412, 1.1111111111111112, 1.130952380952381, 1.1507936507936507, 1.1706349206349205, 1.1904761904761905, 1.2103174603174602, 1.23015873015873, 1.25, 1.2698412698412698, 1.2896825396825395, 1.3095238095238095, 1.3293650793650793, 1.349206349206349, 1.369047619047619, 1.3888888888888888, 1.4087301587301586, 1.4285714285714284, 1.4484126984126984, 1.4682539682539681, 1.488095238095238, 1.507936507936508, 1.5277777777777777, 1.5476190476190474, 1.5674603174603174, 1.5873015873015872, 1.607142857142857, 1.626984126984127, 1.6468253968253967, 1.6666666666666665, 1.6865079365079365, 1.7063492063492063, 1.726190476190476, 1.746031746031746, 1.7658730158730158, 1.7857142857142856, 1.8055555555555554, 1.8253968253968254, 1.8452380952380951, 1.865079365079365, 1.8849206349206349, 1.9047619047619047, 1.9246031746031744, 1.9444444444444444, 1.9642857142857142, 1.984126984126984, 2.003968253968254, 2.0238095238095237, 2.0436507936507935, 2.0634920634920633, 2.083333333333333, 2.1031746031746033, 2.123015873015873, 2.142857142857143, 2.1626984126984126, 2.1825396825396823, 2.202380952380952, 2.2222222222222223, 2.242063492063492, 2.261904761904762, 2.2817460317460316, 2.3015873015873014, 2.321428571428571, 2.341269841269841, 2.361111111111111, 2.380952380952381, 2.4007936507936507, 2.4206349206349205, 2.4404761904761902, 2.46031746031746, 2.4801587301587302, 2.5), #Eta bounds used to set z0 resolution input feature
  associationNetworkZ0ResBins = cms.vdouble(127.0, 126.0, 126.0, 126.0, 125.0, 124.0, 123.0, 122.0, 120.0, 119.0, 117.0, 115.0, 114.0, 112.0, 110.0, 107.0, 105.0, 103.0, 101.0, 98.0, 96.0, 94.0, 91.0, 89.0, 87.0, 85.0, 82.0, 80.0, 78.0, 76.0, 74.0, 72.0, 70.0, 68.0, 66.0, 64.0, 62.0, 61.0, 59.0, 57.0, 56.0, 54.0, 53.0, 51.0, 50.0, 48.0, 47.0, 46.0, 45.0, 43.0, 42.0, 41.0, 40.0, 39.0, 38.0, 37.0, 36.0, 35.0, 34.0, 33.0, 33.0, 32.0, 31.0, 30.0, 30.0, 29.0, 28.0, 28.0, 27.0, 26.0, 26.0, 25.0, 24.0, 24.0, 23.0, 23.0, 22.0, 22.0, 21.0, 21.0, 21.0, 20.0, 20.0, 19.0, 19.0, 18.0, 18.0, 18.0, 17.0, 17.0, 17.0, 16.0, 16.0, 16.0, 15.0, 15.0, 15.0, 15.0, 14.0, 14.0, 14.0, 14.0, 13.0, 13.0, 13.0, 13.0, 12.0, 12.0, 12.0, 12.0, 12.0, 11.0, 11.0, 11.0, 11.0, 11.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 9.0, 9.0, 9.0, 9.0, 9.0, 0.0), #z0 resolution input feature bins
)

l1tTrackVertexAssociationProducerExtended = l1tTrackVertexAssociationProducer.clone(
  l1TracksInputTag = ("l1tGTTInputProducerExtended","Level1TTTracksExtendedConverted"),
  l1SelectedTracksInputTag = cms.InputTag("l1tTrackSelectionProducerExtended", "Level1TTTracksExtendedSelected"),
  l1SelectedTracksEmulationInputTag = cms.InputTag("l1tTrackSelectionProducerExtended", "Level1TTTracksExtendedSelectedEmulation"),
  outputCollectionName = cms.string("Level1TTTracksExtendedSelectedAssociated"),
  cutSet = cms.PSet(
                    #deltaZMaxEtaBounds = cms.vdouble(0.0, absEtaMax.value), # these values define the bin boundaries in |eta|
                    #deltaZMax = cms.vdouble(0.5), # delta z must be less than these values, there will be one less value here than in deltaZMaxEtaBounds, [cm]
                    deltaZMaxEtaBounds = cms.vdouble(0.0, 0.7, 1.0, 1.2, 1.6, 2.0, 2.4), # these values define the bin boundaries in |eta|
                    deltaZMax = cms.vdouble(3.0, 3.0, 3.0, 3.0, 3.0, 3.0), # delta z must be less than these values, there will be one less value here than in deltaZMaxEtaBounds, [cm]
                    ),
  useDisplacedTracksDeltaZOverride = cms.double(3.0), # Use promt/displaced tracks
)

l1tTrackVertexAssociationProducerForJets = l1tTrackVertexAssociationProducer.clone(
  l1SelectedTracksInputTag = cms.InputTag("l1tTrackSelectionProducerForJets", "Level1TTTracksSelected"),
  l1SelectedTracksEmulationInputTag = cms.InputTag("l1tTrackSelectionProducerForJets", "Level1TTTracksSelectedEmulation"),
  cutSet = cms.PSet(
                    #deltaZMaxEtaBounds = cms.vdouble(0.0, absEtaMax.value), # these values define the bin boundaries in |eta|
                    #deltaZMax = cms.vdouble(0.5), # delta z must be less than these values, there will be one less value here than in deltaZMaxEtaBounds, [cm]
                    deltaZMaxEtaBounds = cms.vdouble(0.0, 2.4), # these values define the bin boundaries in |eta|
                    deltaZMax = cms.vdouble(0.55), # delta z must be less than these values, there will be one less value here than in deltaZMaxEtaBounds, [cm]
                    ),
)

l1tTrackVertexAssociationProducerExtendedForJets = l1tTrackVertexAssociationProducerExtended.clone(
  l1SelectedTracksInputTag = cms.InputTag("l1tTrackSelectionProducerExtendedForJets", "Level1TTTracksExtendedSelected"),
  l1SelectedTracksEmulationInputTag = cms.InputTag("l1tTrackSelectionProducerExtendedForJets", "Level1TTTracksExtendedSelectedEmulation"),
  cutSet = cms.PSet(
                    #deltaZMaxEtaBounds = cms.vdouble(0.0, absEtaMax.value), # these values define the bin boundaries in |eta|
                    #deltaZMax = cms.vdouble(0.5), # delta z must be less than these values, there will be one less value here than in deltaZMaxEtaBounds, [cm]
                    deltaZMaxEtaBounds = cms.vdouble(0.0, 2.4), # these values define the bin boundaries in |eta|
                    deltaZMax = cms.vdouble(5.0), # delta z must be less than these values, there will be one less value here than in deltaZMaxEtaBounds, [cm]
                    ),
)

l1tTrackVertexAssociationProducerForEtMiss = l1tTrackVertexAssociationProducer.clone(
  l1SelectedTracksInputTag = cms.InputTag("l1tTrackSelectionProducerForEtMiss", "Level1TTTracksSelected"),
  l1SelectedTracksEmulationInputTag = cms.InputTag("l1tTrackSelectionProducerForEtMiss", "Level1TTTracksSelectedEmulation"),
)

l1tTrackVertexAssociationProducerExtendedForEtMiss = l1tTrackVertexAssociationProducerExtended.clone(
  l1SelectedTracksInputTag = cms.InputTag("l1tTrackSelectionProducerExtendedForEtMiss", "Level1TTTracksExtendedSelected"),
  l1SelectedTracksEmulationInputTag = cms.InputTag("l1tTrackSelectionProducerExtendedForEtMiss", "Level1TTTracksExtendedSelectedEmulation"),
)


