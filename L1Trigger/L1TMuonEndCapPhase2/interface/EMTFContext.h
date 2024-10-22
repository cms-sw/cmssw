#ifndef L1Trigger_L1TMuonEndCapPhase2_EMTFContext_h
#define L1Trigger_L1TMuonEndCapPhase2_EMTFContext_h

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFfwd.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFConfiguration.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFModel.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFTypes.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Algo/HitmapLayer.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Algo/PatternMatchingLayer.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Algo/RoadSortingLayer.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Algo/TrackBuildingLayer.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Algo/DuplicateRemovalLayer.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Algo/ParameterAssignmentLayer.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Algo/OutputLayer.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Data/ActivationLut.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Data/HostLut.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Data/SiteLut.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Data/ZoneLut.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Data/TimeZoneLut.h"

namespace emtf::phase2 {

  // Class
  class EMTFContext {
  public:
    EMTFContext(const edm::ParameterSet&, edm::ConsumesCollector);

    ~EMTFContext();

    // Event configuration
    void update(const edm::Event&, const edm::EventSetup&);

    // Helpers
    GeometryTranslator geometry_translator_;

    // EMTF
    EMTFConfiguration config_;
    EMTFModel model_;

    // Prompt Neural Network
    tensorflow::GraphDef* prompt_graph_ptr_;
    tensorflow::Session* prompt_session_ptr_;

    // Displaced Neural Network
    tensorflow::GraphDef* disp_graph_ptr_;
    tensorflow::Session* disp_session_ptr_;

    // Data
    data::SiteLut site_lut_;
    data::HostLut host_lut_;
    data::ZoneLut zone_lut_;
    data::TimeZoneLut timezone_lut_;
    data::ActivationLut activation_lut_;

    // Algorithm
    algo::HitmapLayer hitmap_building_layer_;
    algo::PatternMatchingLayer pattern_matching_layer_;
    algo::RoadSortingLayer road_sorting_layer_;
    algo::TrackBuildingLayer track_building_layer_;
    algo::DuplicateRemovalLayer duplicate_removal_layer_;
    algo::ParameterAssignmentLayer parameter_assignment_layer_;
    algo::OutputLayer output_layer_;
  };

}  // namespace emtf::phase2

#endif  // L1Trigger_L1TMuonEndCapPhase2_EMTFContext_h not defined
