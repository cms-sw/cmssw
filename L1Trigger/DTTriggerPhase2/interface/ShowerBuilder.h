#ifndef Phase2L1Trigger_DTTrigger_ShowerBuilder_h
#define Phase2L1Trigger_DTTrigger_ShowerBuilder_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "L1Trigger/DTTriggerPhase2/interface/ShowerCandidate.h"
#include "L1Trigger/DTTriggerPhase2/interface/constants.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"

#include <iostream>
#include <fstream>

#include "TTree.h"

// ===============================================================================
// Previous definitions and declarations
// ===============================================================================

namespace showerb {
  typedef std::pair<int, DTPrimitive> DTPrimPlusBx;
  typedef std::deque<DTPrimPlusBx> ShowerBuffer;

  inline bool hitWireSort_shower(const DTPrimitive& hit1, const DTPrimitive& hit2) {
    int wi1 = hit1.channelId();
    int wi2 = hit2.channelId();

    if (wi1 < wi2)
      return true;
    else
      return false;
  }

  inline bool hitLayerSort_shower(const DTPrimitive& hit1, const DTPrimitive& hit2) {
    int lay1 = hit1.layerId();
    int lay2 = hit2.layerId();

    if (lay1 < lay2)
      return true;
    else if (lay1 > lay2)
      return false;
    else
      return hitWireSort_shower(hit1, hit2);
  }

  inline bool hitTimeSort_shower(const DTPrimitive& hit1, const DTPrimitive& hit2) {
    int tdc1 = hit1.tdcTimeStamp();
    int tdc2 = hit2.tdcTimeStamp();

    if (tdc1 < tdc2)
      return true;
    else
      return false;
    // else if (tdc1 > tdc2) return false;
    // else return hitLayerSort_shower(hit1, hit2); --> ignoring those sortings for now
  }

  inline float compute_avg_pos(DTPrimitives& hits) {
    int nhits_ = hits.size();

    if (nhits_ == 0)
      return -1.0;

    float aux_avgPos_ = 0;

    for (auto& hit : hits) {
      aux_avgPos_ += hit.wireHorizPos();
    }

    return aux_avgPos_ / nhits_;
  }

  inline float compute_avg_time(DTPrimitives& hits) {
    int nhits_ = hits.size();

    if (nhits_ == 0)
      return -1.0;

    float aux_avgTime_ = 0;

    for (auto& hit : hits) {
      aux_avgTime_ += hit.tdcTimeStamp();
    }
    return aux_avgTime_ / nhits_;
  }

  inline void set_wire_properties(ShowerCandidatePtr& shower_cand, DTPrimitives& hits) {
    auto& wires_profile = shower_cand->getWiresProfile();
    auto& wires_constituents = shower_cand->getWiresConstituents();
    auto& wires_layer_constituents = shower_cand->getWiresLayerConstituents();
    auto& wires_tdc_constituents = shower_cand->getWiresTdcConstituents();

    for (auto& hit : hits) {
      wires_profile[hit.channelId() - 1]++;
      wires_constituents.push_back(hit.channelId());
      wires_layer_constituents.push_back(hit.layerId());
      wires_tdc_constituents.push_back(hit.tdcTimeStamp());
    }
  }

  inline bool buffer_contains(const ShowerBuffer& buffer, const DTPrimitive& hit) {
    for (const auto& item : buffer) {
      if (item.second.channelId() == hit.channelId() && item.second.layerId() == hit.layerId() &&
          item.second.superLayerId() == hit.superLayerId()) {
        return true;
      }
    }
    return false;
  }

  inline void buffer_get_hits(const ShowerBuffer& buffer, DTPrimitives& hits) {
    for (const auto& item : buffer) {
      hits.push_back(item.second);
    }
  }

  inline void buffer_clear_olds(ShowerBuffer& buffer, const int _current_bx, const int persistence_bx_units) {
    while (!buffer.empty() && (_current_bx - buffer.front().first) > persistence_bx_units) {
      buffer.pop_front();
    }
  }

  inline void buffer_reset(ShowerBuffer& buffer) { buffer.clear(); }

  inline std::string buffer_to_string(const ShowerBuffer& buffer) {
    // Create a string representation of the buffer contents
    // Format: [BX:SL:Layer:Wire, BX:SL:Layer:Wire, ...]
    std::string contents = "[";
    for (const auto& hit : buffer) {
      if (contents.length() > 1)
        contents += ", ";
      contents += std::to_string(hit.first) + ":" + std::to_string(hit.second.superLayerId()) + ":" +
                  std::to_string(hit.second.layerId()) + ":" + std::to_string(hit.second.channelId());
    }
    contents += "]";
    return contents;
  }

  // Stream-like debug logger class
  class DebugLogger {
  public:
    DebugLogger(const std::string& category) : category_(category) {}

    // Template to accept any type with << operator
    template <typename T>
    DebugLogger& operator<<(const T& value) {
      stream_ << value;
      return *this;
    }

    // Handle std::endl and other manipulators
    DebugLogger& operator<<(std::ostream& (*manip)(std::ostream&)) {
      stream_ << manip;
      return *this;
    }

    // Destructor - actually performs the logging
    ~DebugLogger() {
      LogDebug(category_) << stream_.str();
      std::cout << stream_.str() << std::endl;
    }

  private:
    std::string category_;
    std::ostringstream stream_;
  };

  // Helper function to create the logger
  inline DebugLogger log_debug(const std::string& category) { return DebugLogger(category); }
}  // namespace showerb

// ===============================================================================
// Class declarations
// ===============================================================================

class ShowerBuilder {
public:
  // Constructors and destructor
  ShowerBuilder(const edm::ParameterSet& pset,
                edm::ConsumesCollector& iC,
                TTree* tree);  // Changed from std::shared_ptr<TTree> to TTree*

  // Main methods
  void initialise(const edm::EventSetup& iEventSetup);
  void run(edm::Event& iEvent,
           const edm::EventSetup& iEventSetup,
           const DTDigiCollection& digis,
           ShowerCandidatesMap& dtshowers_out,
           const DTChamber* chamber);

private:
  // Private auxiliary methods
  void clear();
  void setInChannels(const DTDigiCollection* digi);
  void processHits_standAlone();
  void processHitsFirmwareEmulation();

  bool triggerShower(const showerb::ShowerBuffer& buffer);
  void set_shower_properties(ShowerCandidatePtr& showerCand,
                             showerb::ShowerBuffer& buffer,
                             int nhits = -1,
                             int min_wire = -1,
                             int max_wire = -1,
                             float avg_pos = -1.,
                             float avg_time = -1.);
  void groupHits_byBx();
  void fill_obdt(const int bx);
  void fill_bmtl1_buffers(const int bx);
  void bxStep(const int _current_bx);
  void dump_digi_to_tree(showerb::DTPrimPlusBx& hitpbx);

  // Private attributes
  const int showerTaggingAlgo_;
  const int threshold_for_shower_;
  const int nHits_per_bx_phi_;
  const int nHits_per_bx_z_;
  const int obdt_hits_bxpersistence_phi_;
  const int obdt_hits_bxpersistence_z_;
  const int obdt_wire_relaxing_time_;
  const int bmtl1_hits_bxpersistence_;
  const bool debug_;
  const bool dump_digis_;
  const int scenario_;
  int bx_shift_back_;
  int time_shift_back_;

  // auxiliary variables
  DTPrimitives all_hits;
  std::map<int, DTPrimitives, std::less<int>> all_hits_perBx;

  std::map<int, ShowerCandidatePtrs> showerCands;  // defined as a map to easy acces with SL number

  // initialize buffers
  showerb::ShowerBuffer obdt_buffer_phi_;       // Buffer to emulate the OBDT behavior
  showerb::ShowerBuffer obdt_buffer_z_;         // Buffer to emulate the OBDT behavior
  showerb::ShowerBuffer hot_wires_buffer_phi_;  // Buffer to emulate the hot wires behavior in OBDT-phi
  showerb::ShowerBuffer hot_wires_buffer_z_;    // Buffer to emulate the hot wires behavior in OBDT-z

  std::map<int, std::pair<showerb::ShowerBuffer*, showerb::ShowerBuffer*>> obdt_buffers = {
      {1, {&obdt_buffer_phi_, &hot_wires_buffer_phi_}},
      {2, {&obdt_buffer_z_, &hot_wires_buffer_z_}},
      {3, {&obdt_buffer_phi_, &hot_wires_buffer_phi_}}  // 1 and 3 point to the same buffers
  };
  showerb::ShowerBuffer bmtl1_sl1_buffer_;  // Buffer to emulate the BMTL1 behavior for SL1
  showerb::ShowerBuffer bmtl1_sl3_buffer_;  // Buffer to emulate the BMTL1 behavior for SL3
  showerb::ShowerBuffer bmtl1_sl2_buffer_;  // Buffer to emulate the BMTL1 behavior for SL2

  std::map<int, showerb::ShowerBuffer*> bmtl1_buffers = {  // Buffers to emulate the BMTL1 shower buffer SL1, SL2, SL3
      {1, &bmtl1_sl1_buffer_},
      {3, &bmtl1_sl3_buffer_},
      {2, &bmtl1_sl2_buffer_}};

  // To dump digis in case requested
  TTree* m_tree;

  int m_event_number;
  std::vector<short> m_hit_wheel;
  std::vector<short> m_hit_sector;
  std::vector<short> m_hit_station;
  std::vector<short> m_hit_superlayer;
  std::vector<short> m_hit_layer;
  std::vector<int> m_hit_wire;
  std::vector<float> m_hit_tdc;
  std::vector<int> m_hit_bx;
};

#endif
