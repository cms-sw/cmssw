#include <string>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFConfiguration.h"

using namespace emtf::phase2;

EMTFConfiguration::EMTFConfiguration(const edm::ParameterSet& pset) {
  verbosity_ = pset.getUntrackedParameter<int>("Verbosity");

  // Validation
  validation_dir_ = pset.getParameter<std::string>("ValidationDirectory");

  // Neural Network
  prompt_graph_path_ = pset.getParameter<std::string>("PromptGraphPath");
  displ_graph_path_ = pset.getParameter<std::string>("DisplacedGraphPath");

  // Trigger
  min_bx_ = pset.getParameter<int>("MinBX");
  max_bx_ = pset.getParameter<int>("MaxBX");
  bx_window_ = pset.getParameter<int>("BXWindow");

  // Subsystems
  csc_en_ = pset.getParameter<bool>("CSCEnabled");
  rpc_en_ = pset.getParameter<bool>("RPCEnabled");
  gem_en_ = pset.getParameter<bool>("GEMEnabled");
  me0_en_ = pset.getParameter<bool>("ME0Enabled");
  ge0_en_ = pset.getParameter<bool>("GE0Enabled");

  csc_bx_shift_ = pset.getParameter<int>("CSCInputBXShift");
  rpc_bx_shift_ = pset.getParameter<int>("RPCInputBXShift");
  gem_bx_shift_ = pset.getParameter<int>("GEMInputBXShift");
  me0_bx_shift_ = pset.getParameter<int>("ME0InputBXShift");

  csc_input_ = pset.getParameter<edm::InputTag>("CSCInput");
  rpc_input_ = pset.getParameter<edm::InputTag>("RPCInput");
  gem_input_ = pset.getParameter<edm::InputTag>("GEMInput");
  me0_input_ = pset.getParameter<edm::InputTag>("ME0Input");
  ge0_input_ = pset.getParameter<edm::InputTag>("GE0Input");

  // Primitive Selection
  include_neighbor_en_ = pset.getParameter<bool>("IncludeNeighborEnabled");
}

EMTFConfiguration::~EMTFConfiguration() {}

void EMTFConfiguration::update(const edm::Event& i_event, const edm::EventSetup& i_event_setup) {
  // Do Nothing
}
