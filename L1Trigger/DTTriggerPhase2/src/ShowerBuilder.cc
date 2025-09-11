#include "L1Trigger/DTTriggerPhase2/interface/ShowerCandidate.h"
#include "L1Trigger/DTTriggerPhase2/interface/ShowerBuilder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace cmsdt;

// ============================================================================
// Constructors and destructor
// ============================================================================
ShowerBuilder::ShowerBuilder(const edm::ParameterSet &pset, edm::ConsumesCollector &iC, TTree *tree)
    :  // Unpack information from pset
      showerTaggingAlgo_(pset.getParameter<int>("showerTaggingAlgo")),
      threshold_for_shower_(pset.getParameter<int>("threshold_for_shower")),
      nHits_per_bx_phi_(pset.getParameter<int>("nHits_per_bx_phi")),
      nHits_per_bx_z_(pset.getParameter<int>("nHits_per_bx_z")),
      obdt_hits_bxpersistence_phi_(pset.getParameter<int>("obdt_hits_bxpersistence_phi")),
      obdt_hits_bxpersistence_z_(pset.getParameter<int>("obdt_hits_bxpersistence_z")),
      obdt_wire_relaxing_time_(pset.getParameter<int>("obdt_wire_relaxing_time")),
      bmtl1_hits_bxpersistence_(pset.getParameter<int>("bmtl1_hits_bxpersistence")),
      debug_(pset.getUntrackedParameter<bool>("debug")),
      dump_digis_(pset.getUntrackedParameter<bool>("dump_digis")),
      scenario_(pset.getParameter<int>("scenario")),
      m_tree(tree) {
  int bx_shift = 0;
  if (scenario_ == MC)         //scope for MC
    bx_shift = 400;            // value used in standard CMSSW simulation
  else if (scenario_ == DATA)  //scope for data
    bx_shift = 0;
  else if (scenario_ == SLICE_TEST)  //scope for slice test
    bx_shift = 400;                  // slice test to mimic simulation

  bx_shift_back_ = bx_shift;
  time_shift_back_ = bx_shift_back_ * 25;
  if (dump_digis_) {
    // Book tree branches
    m_tree->Branch("event_eventNumber", &m_event_number);
    m_tree->Branch("wheel", &m_hit_wheel);
    m_tree->Branch("sector", &m_hit_sector);
    m_tree->Branch("station", &m_hit_station);
    m_tree->Branch("superlayer", &m_hit_superlayer);
    m_tree->Branch("layer", &m_hit_layer);
    m_tree->Branch("wire", &m_hit_wire);
    m_tree->Branch("tdc", &m_hit_tdc);
    m_tree->Branch("bx", &m_hit_bx);
  }
}

// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void ShowerBuilder::initialise(const edm::EventSetup &iEventSetup) {
  // clear branch vectors - once per event
  if (dump_digis_) {
    m_hit_wheel.clear();
    m_hit_sector.clear();
    m_hit_station.clear();
    m_hit_superlayer.clear();
    m_hit_layer.clear();
    m_hit_wire.clear();
    m_hit_tdc.clear();
    m_hit_bx.clear();
  }
}

void ShowerBuilder::run(edm::Event &iEvent,
                        const edm::EventSetup &iEventSetup,
                        const DTDigiCollection &digis,
                        ShowerCandidatesMap &dtshowers_out,
                        const DTChamber *chamber) {
  m_event_number = iEvent.id().event();
  // Clear auxiliars
  clear();
  // Set the incoming hits in the channels
  setInChannels(&digis);

  int nHits = all_hits.size();
  if (nHits != 0) {
    if (debug_) {
      showerb::log_debug("ShowerBuilder") << "        - Going to study " << nHits << " hits";
      for (auto &cand : showerCands) {
        showerb::log_debug("ShowerBuilder")
            << "        - Candidate vector for SL" << cand.first << " initialized with size " << cand.second.size();
      }
    }
    if (showerTaggingAlgo_ == 0) {
      // Standalone mode: just save hits and flag if the number of hits is above the threshold
      processHits_standAlone();
    } else if (showerTaggingAlgo_ == 1) {
      // Firmware emulation:
      // mimics the behavior of sending and receiving hits from the OBDT to the shower algorithm in the BMTL1.
      processHitsFirmwareEmulation();
    }
  } else {
    if (debug_)
      showerb::log_debug("ShowerBuilder") << "        - No hits to study.";
  }

  for (int sl : {1, 2, 3}) {
    // MB4 stations does not have SL2, so check it first
    if (chamber->superLayer(sl) && !showerCands[sl].empty()) {
      DTSuperLayerId sl1id = chamber->superLayer(sl)->id();
      dtshowers_out[sl1id] = std::move(showerCands[sl]);
    }
  }
}

// ============================================================================
// Auxiliary methods
// ============================================================================
void ShowerBuilder::clear() {
  all_hits.clear();
  all_hits_perBx.clear();
  // clear buffers
  for (auto &pair : obdt_buffers) {
    showerb::buffer_reset(*pair.second.first);
    showerb::buffer_reset(*pair.second.second);
  }
  for (auto &pair : bmtl1_buffers) {
    showerb::buffer_reset(*pair.second);
  }
  // recreated candidates each time since the results are moved out each run execution
  showerCands = {{1, ShowerCandidatePtrs{}}, {3, ShowerCandidatePtrs{}}, {2, ShowerCandidatePtrs{}}};
}

void ShowerBuilder::setInChannels(const DTDigiCollection *digis) {
  for (const auto &dtLayerId_It : *digis) {
    const DTLayerId id = dtLayerId_It.first;
    // iterate over the digis
    for (DTDigiCollection::const_iterator digiIt = (dtLayerId_It.second).first; digiIt != (dtLayerId_It.second).second;
         ++digiIt) {
      auto dtpAux = DTPrimitive();

      dtpAux.setTDCTimeStamp((*digiIt).time());
      dtpAux.setTimeCorrection(time_shift_back_);
      dtpAux.setOrbit((*digiIt).countsTDC() / (*digiIt).tdcBase());
      dtpAux.setOrbitCorrection(bx_shift_back_);
      dtpAux.setChannelId((*digiIt).wire());

      dtpAux.setLayerId(id.layer());
      dtpAux.setSuperLayerId(id.superlayer());
      dtpAux.setCameraId(id.rawId());

      all_hits.push_back(dtpAux);
    }
  }
  // sort the hits by time
  std::stable_sort(all_hits.begin(), all_hits.end(), showerb::hitTimeSort_shower);
}

void ShowerBuilder::processHits_standAlone() {
  // Standalone mode: for each superlayer, fill the buffer and check if nHits >= threshold
  std::map<int, ShowerCandidatePtr> candidates = {// only three  Candidates are possible in this case
                                                  {1, std::make_shared<ShowerCandidate>()},
                                                  {2, std::make_shared<ShowerCandidate>()},
                                                  {3, std::make_shared<ShowerCandidate>()}};

  // Fill buffers
  for (auto &hit : all_hits) {
    int sl = hit.superLayerId();
    bmtl1_buffers[sl]->push_back(showerb::DTPrimPlusBx(-1, hit));
  }
  // Process each superlayer
  for (int sl : {1, 2, 3}) {
    if (triggerShower(*bmtl1_buffers[sl])) {
      if (debug_)
        showerb::log_debug("ShowerBuilder")
            << "        o Shower found in SL" << sl << " with " << bmtl1_buffers[sl]->size() << " hits";
      candidates[sl]->flag();
      set_shower_properties(candidates[sl], *bmtl1_buffers[sl]);
    }
    showerCands[sl].push_back(candidates[sl]);
  }
}

void ShowerBuilder::processHitsFirmwareEmulation() {
  // Use a for over BXs to emulate the behavior of the OBDT and BMTL1, this means considering:
  // 1. OBDT can only store recived hits during 4 BXs (adjustable with obdt_hits_bxpersistence_phi/z_)
  // 2. OBDT can not recive hits from the same wire during 2 BXs (adjustable with obdt_wire_relaxing_time_)
  // 3. OBDT can only sends 8 hits (less if OBDT-z) per BX to the BMTL1 (adjustable with obdt_hits_per_bx_phi/z)
  // 4. Shower algorithm in the BMTL1 mantains the hits along 16 BXs and triggers if nHits >= threshold (adjustable with bmtl1_hits_bxpersistence_)
  groupHits_byBx();

  // auxiliary variables
  std::map<int, int> prev_nHits = {{1, 0}, {2, 0}, {3, 0}};
  std::map<int, int> nHits = {{1, 0}, {2, 0}, {3, 0}};
  std::map<int, bool> shower_already_set = {{1, false}, {2, false}, {3, false}};
  // prepare the candidates with at least one entry for each superlayer
  for (int sl : {1, 2, 3}) {
    showerCands[sl].push_back(std::make_shared<ShowerCandidate>());
  }

  // necessary to don't lose the first hits in BMTL1 buffers
  std::map<int, showerb::ShowerBuffer> prev_bmtl1_buffers_state_;

  // -------------------

  int min_bx = all_hits_perBx.begin()->first;
  int max_bx = all_hits_perBx.rbegin()->first;

  if (debug_)
    showerb::log_debug("ShowerBuilder") << "        - bx range: " << min_bx << " - " << max_bx
                                        << ", size: " << (max_bx - min_bx);

  for (int bx = min_bx; bx <= max_bx + 17; bx++) {
    // Clear old hits from buffers
    bxStep(bx);

    fill_obdt(bx);

    if (debug_) {
      for (int sl : {1, 2}) {
        showerb::log_debug("ShowerBuilder") << "          ^ OBDT buffer for SL" << sl << " at BX " << bx << ": "
                                            << showerb::buffer_to_string(*obdt_buffers[sl].first);
        showerb::log_debug("ShowerBuilder") << "          ^ Hot wires buffer for SL" << sl << " at BX " << bx << ": "
                                            << showerb::buffer_to_string(*obdt_buffers[sl].second);
      }
    }
    fill_bmtl1_buffers(bx);

    // For each superlayer, update nHits, print debug, check trigger, and set shower properties
    for (int sl : {1, 2, 3}) {
      nHits[sl] = bmtl1_buffers[sl]->size();

      if (debug_)
        showerb::log_debug("ShowerBuilder") << "          ^ bmtl1 buffer for SL" << sl << " at BX " << bx << ": "
                                            << showerb::buffer_to_string(*bmtl1_buffers[sl]);

      // Operate on last candidate in vector
      ShowerCandidatePtr &cand = showerCands[sl].back();
      // Trigger shower and flag
      if (triggerShower(*bmtl1_buffers[sl]) && !cand->isFlagged()) {
        if (debug_)
          showerb::log_debug("ShowerBuilder")
              << "        o Shower found in SL" << sl << " with " << nHits[sl] << " hits";
        cand->flag();
      }
      // Set shower properties if needed
      if (nHits[sl] < prev_nHits[sl] && cand->isFlagged() && !shower_already_set[sl]) {
        shower_already_set[sl] = true;
        set_shower_properties(cand, prev_bmtl1_buffers_state_[sl], prev_nHits[sl]);
      } else {
        prev_nHits[sl] = nHits[sl];
      }
      if (cand->isFlagged() && !shower_already_set[sl]) {
        // Save current buffer state for next iteration, ONLY for flagged showers that haven't been set yet
        prev_bmtl1_buffers_state_[sl] = *bmtl1_buffers[sl];
      }
    }
  }
}

bool ShowerBuilder::triggerShower(const showerb::ShowerBuffer &buffer) {
  int nHits = buffer.size();
  if (nHits >= threshold_for_shower_) {
    return true;
  }
  return false;
}

void ShowerBuilder::set_shower_properties(ShowerCandidatePtr &showerCand,
                                          showerb::ShowerBuffer &buffer,
                                          int nhits,
                                          int min_wire,
                                          int max_wire,
                                          float avg_pos,
                                          float avg_time) {
  showerCand->setBX(buffer.front().first);

  DTPrimitives _hits;

  showerb::buffer_get_hits(buffer, _hits);
  std::stable_sort(_hits.begin(), _hits.end(), showerb::hitWireSort_shower);

  // change values considering the buffer or the input values
  nhits = (nhits == -1) ? _hits.size() : nhits;
  min_wire = (min_wire == -1) ? _hits.front().channelId() : min_wire;
  max_wire = (max_wire == -1) ? _hits.back().channelId() : max_wire;
  avg_pos = (avg_pos == -1) ? showerb::compute_avg_pos(_hits) : avg_pos;
  avg_time = (avg_time == -1) ? showerb::compute_avg_time(_hits) : avg_time;

  showerCand->setNhits(nhits);
  showerCand->setMinWire(min_wire);
  showerCand->setMaxWire(max_wire);
  showerCand->setAvgPos(avg_pos);
  showerCand->setAvgTime(avg_time);

  showerb::set_wire_properties(showerCand, _hits);

  if (debug_) {
    showerb::log_debug("ShowerBuilder") << "        o Setting shower properties with buffer: "
                                        << showerb::buffer_to_string(buffer);
    showerb::log_debug("ShowerBuilder") << "          - Setting BX: " << buffer.front().first;
    showerb::log_debug("ShowerBuilder") << "          - Setting nhits: " << nhits;
    showerb::log_debug("ShowerBuilder") << "          - Setting min_wire: " << min_wire;
    showerb::log_debug("ShowerBuilder") << "          - Setting max_wire: " << max_wire;
    showerb::log_debug("ShowerBuilder") << "          - Setting avg_pos: " << avg_pos;
    showerb::log_debug("ShowerBuilder") << "          - Setting avg_time: " << avg_time;
    showerb::log_debug("ShowerBuilder") << "          - Wire properties set successfully";
  }
}

void ShowerBuilder::groupHits_byBx() {
  all_hits_perBx.clear();
  // Group hits by BX
  for (auto &hit : all_hits) {
    // Compute the BX from the TDC time
    int bx = hit.orbitNoOffset();
    all_hits_perBx[bx].push_back(hit);
  }
}

void ShowerBuilder::fill_obdt(const int bx) {
  // Fill the OBDT buffer with the hits in the current BX

  if (all_hits_perBx.find(bx) == all_hits_perBx.end()) {
    return;
  }  // if there are not hits in this BX nothing to do

  for (auto &hit : all_hits_perBx[bx]) {
    int sl = hit.superLayerId();
    auto &obdt_buffer = *obdt_buffers[sl].first;
    auto &hot_wires_buffer = *obdt_buffers[sl].second;
    if (debug_)
      showerb::log_debug("ShowerBuilder") << "          ^ Trying to add hit with wire " << hit.layerId() << ":"
                                          << hit.channelId() << " in OBDT sl:" << sl << " at BX " << bx;

    if (!showerb::buffer_contains(hot_wires_buffer, hit)) {
      if (debug_)
        showerb::log_debug("ShowerBuilder") << "          ^ added";

      showerb::DTPrimPlusBx _hit(bx, hit);
      obdt_buffer.push_back(_hit);
      hot_wires_buffer.push_back(_hit);
    }
  }
}

void ShowerBuilder::fill_bmtl1_buffers(const int bx) {
  // Fill the BMTL1 buffer with the hits in the OBDT buffer only nHits_per_bx_phi/z_ hits are added

  // if all OBDT buffers are empty, nothing to do
  bool all_empty = true;
  for (int sl : {1, 2}) {
    if (!obdt_buffers[sl].first->empty()) {
      all_empty = false;
      break;
    }
  }
  if (all_empty)
    return;

  if (debug_)
    showerb::log_debug("ShowerBuilder") << "          ^ Getting hits from OBDT";

  std::map<int, int> hits_per_bx = {{1, nHits_per_bx_phi_}, {2, nHits_per_bx_z_}, {3, nHits_per_bx_phi_}};
  int hits_count_phi_ = 0;
  int hits_count_z_ = 0;
  std::map<int, int *> hits_count = {{1, &hits_count_phi_}, {2, &hits_count_z_}, {3, &hits_count_phi_}};

  for (const auto &pair : obdt_buffers) {
    auto &obdt_buffer = *pair.second.first;
    while (!obdt_buffer.empty()) {  // if empty, nothing to do
      auto _hitpbx = obdt_buffer.front();
      int sl = _hitpbx.second.superLayerId();
      if (*hits_count[sl] >= hits_per_bx[sl])  // don't add more hits if the max per bx is reached
        break;
      _hitpbx.first = bx;  // set the correct BX in the hit in the context of the BMTL1 (BXsend)
      bmtl1_buffers[sl]->push_back(_hitpbx);
      (*hits_count[sl])++;
      if (dump_digis_)
        dump_digi_to_tree(_hitpbx);
      obdt_buffer.pop_front();
    }
  }
}

void ShowerBuilder::bxStep(const int _current_bx) {
  // Remove old elements from the buffers
  std::map<int, int> obdt_hits_bxpersistences = {
      {1, obdt_hits_bxpersistence_phi_}, {3, obdt_hits_bxpersistence_phi_}, {2, obdt_hits_bxpersistence_z_}};
  for (auto &pair : obdt_buffers) {
    int sl = pair.first;
    showerb::buffer_clear_olds(*pair.second.first, _current_bx, obdt_hits_bxpersistences[sl]);
    showerb::buffer_clear_olds(*pair.second.second, _current_bx, obdt_wire_relaxing_time_);
  }

  for (auto &pair : bmtl1_buffers) {
    showerb::buffer_clear_olds(*pair.second, _current_bx, bmtl1_hits_bxpersistence_);
  }
}

void ShowerBuilder::dump_digi_to_tree(showerb::DTPrimPlusBx &hitpbx) {
  DTPrimitive &hit = hitpbx.second;

  DTChamberId chId(hit.cameraId());

  m_hit_wheel.push_back(chId.wheel());
  m_hit_sector.push_back(chId.sector());
  m_hit_station.push_back(chId.station());
  m_hit_superlayer.push_back(hit.superLayerId());
  m_hit_layer.push_back(hit.layerId());
  m_hit_wire.push_back(hit.channelId());
  m_hit_tdc.push_back(hit.tdcTimeStampNoOffset());
  int bxsend = hitpbx.first;
  m_hit_bx.push_back(bxsend);
}
