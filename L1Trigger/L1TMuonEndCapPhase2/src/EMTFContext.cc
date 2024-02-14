#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "L1Trigger/L1TMuon/interface/GeometryTranslator.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFConfiguration.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFContext.h"

using namespace emtf::phase2;

EMTFContext::EMTFContext(const edm::ParameterSet& pset, edm::ConsumesCollector i_consumes_collector)
    :  // Helpers
      geometry_translator_(i_consumes_collector),

      // EMTF
      config_(pset),
      model_(*this),

      // Prompt Neural Network
      prompt_graph_ptr_(nullptr),
      prompt_session_ptr_(nullptr),

      // Displaced Neural Network
      disp_graph_ptr_(nullptr),
      disp_session_ptr_(nullptr),

      // Data
      site_lut_(),
      host_lut_(),
      zone_lut_(),
      timezone_lut_(),
      activation_lut_(),

      // Layers
      hitmap_building_layer_(*this),
      pattern_matching_layer_(*this),
      road_sorting_layer_(*this),
      track_building_layer_(*this),
      duplicate_removal_layer_(*this),
      parameter_assignment_layer_(*this),
      output_layer_(*this) {
  // Do Nothing
}

EMTFContext::~EMTFContext() {
  // Delete Prompt Neural Network
  if (prompt_session_ptr_ != nullptr) {
    tensorflow::closeSession(prompt_session_ptr_);
    delete prompt_session_ptr_;
  }

  if (prompt_graph_ptr_ != nullptr) {
    delete prompt_graph_ptr_;
  }

  // Delete Displaced Neural Network
  if (disp_session_ptr_ != nullptr) {
    tensorflow::closeSession(disp_session_ptr_);
    delete disp_session_ptr_;
  }

  if (disp_graph_ptr_ != nullptr) {
    delete disp_graph_ptr_;
  }
}

void EMTFContext::update(const edm::Event& i_event, const edm::EventSetup& i_event_setup) {
  // Update Helpers
  geometry_translator_.checkAndUpdateGeometry(i_event_setup);

  // Update Config
  config_.update(i_event, i_event_setup);

  // Update Prompt Neural Network
  if (prompt_session_ptr_ != nullptr) {
    delete prompt_session_ptr_;
  }

  if (prompt_graph_ptr_ != nullptr) {
    delete prompt_graph_ptr_;
  }

  prompt_graph_ptr_ = tensorflow::loadGraphDef(edm::FileInPath(config_.prompt_graph_path_).fullPath());

  prompt_session_ptr_ = tensorflow::createSession(prompt_graph_ptr_);

  // Update Displaced Neural Network
  if (disp_session_ptr_ != nullptr) {
    delete disp_session_ptr_;
  }

  if (disp_graph_ptr_ != nullptr) {
    delete disp_graph_ptr_;
  }

  disp_graph_ptr_ = tensorflow::loadGraphDef(edm::FileInPath(config_.displ_graph_path_).fullPath());

  disp_session_ptr_ = tensorflow::createSession(disp_graph_ptr_);

  // Update Data
  site_lut_.update(i_event, i_event_setup);
  host_lut_.update(i_event, i_event_setup);
  zone_lut_.update(i_event, i_event_setup);
  timezone_lut_.update(i_event, i_event_setup);
  activation_lut_.update(i_event, i_event_setup);
}
