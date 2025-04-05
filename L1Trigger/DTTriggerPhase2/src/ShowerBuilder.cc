#include "L1Trigger/DTTriggerPhase2/interface/ShowerCandidate.h"
#include "L1Trigger/DTTriggerPhase2/interface/ShowerBuilder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace edm;
using namespace std;
using namespace cmsdt;

// ============================================================================
// Constructors and destructor
// ============================================================================
ShowerBuilder::ShowerBuilder(const ParameterSet &pset, edm::ConsumesCollector &iC)
    :  // Unpack information from pset
      showerTaggingAlgo_(pset.getParameter<int>("showerTaggingAlgo")),
      threshold_for_shower_(pset.getParameter<int>("threshold_for_shower")),
      nHits_per_bx_(pset.getParameter<int>("nHits_per_bx")),
      obdt_hits_bxpersistence_(pset.getParameter<int>("obdt_hits_bxpersistence")),
      obdt_wire_relaxing_time_(pset.getParameter<int>("obdt_wire_relaxing_time")),
      bmtl1_hits_bxpersistence_(pset.getParameter<int>("bmtl1_hits_bxpersistence")),
      debug_(pset.getUntrackedParameter<bool>("debug")),
      scenario_(pset.getParameter<int>("scenario")) {}

ShowerBuilder::~ShowerBuilder() {}

// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void ShowerBuilder::initialise(const edm::EventSetup &iEventSetup) {}

void ShowerBuilder::run(Event &iEvent,
                        const EventSetup &iEventSetup,
                        const DTDigiCollection &digis,
                        ShowerCandidatePtr &showerCandidate_SL1,
                        ShowerCandidatePtr &showerCandidate_SL3) {
  // Clear auxiliars
  clear();
  // Set the incoming hits in the channels
  setInChannels(&digis);

  std::map<int, ShowerCandidatePtr> aux_showerCands{// defined as a map to easy acces with SL number
                                                    {1, make_shared<ShowerCandidate>()},
                                                    {3, make_shared<ShowerCandidate>()}};

  int nHits = all_hits.size();
  if (nHits != 0) {
    if (debug_)
      LogDebug("ShowerBuilder") << "        - Going to study " << nHits << " hits";
    if (showerTaggingAlgo_ == 0) {
      // Standalone mode: just save hits and flag if the number of hits is above the threshold
      processHits_standAlone(aux_showerCands);
    } else if (showerTaggingAlgo_ == 1) {
      // Firmware emulation:
      // mimics the behavior of sending and receiving hits from the OBDT to the shower algorithm in the BMTL1.
      processHitsFirmwareEmulation(aux_showerCands);
    }
  } else {
    if (debug_)
      LogDebug("ShowerBuilder") << "        - No hits to study.";
  }

  showerCandidate_SL1 = std::move(aux_showerCands[1]);
  showerCandidate_SL3 = std::move(aux_showerCands[3]);
}

void ShowerBuilder::finish() {};

// ============================================================================
// Auxiliary methods
// ============================================================================
void ShowerBuilder::clear() {
  all_hits.clear();
  all_hits_perBx.clear();
  showerb::buffer_reset(obdt_buffer);
  showerb::buffer_reset(hot_wires_buffer);
  showerb::buffer_reset(bmtl1_sl1_buffer);
  showerb::buffer_reset(bmtl1_sl3_buffer);
}

void ShowerBuilder::setInChannels(const DTDigiCollection *digis) {
  for (const auto &dtLayerId_It : *digis) {
    const DTLayerId id = dtLayerId_It.first;
    if (id.superlayer() == 2)
      continue;  // Skip SL2 digis
    // Now iterate over the digis
    for (DTDigiCollection::const_iterator digiIt = (dtLayerId_It.second).first; digiIt != (dtLayerId_It.second).second;
         ++digiIt) {
      auto dtpAux = DTPrimitive();
      dtpAux.setTDCTimeStamp((*digiIt).time());
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

void ShowerBuilder::processHits_standAlone(std::map<int, ShowerCandidatePtr> &showerCands) {
  // For each superlayer, fill the buffer and check if nHits >= threshold
  for (auto &hit : all_hits) {
    showerb::DTPrimPlusBx _hitpbx(-1, hit);
    if (hit.superLayerId() == 1)
      bmtl1_sl1_buffer.push_back(_hitpbx);
    else if (hit.superLayerId() == 3)
      bmtl1_sl3_buffer.push_back(_hitpbx);
  }

  if (triggerShower(bmtl1_sl1_buffer)) {
    if (debug_) {
      int nHits_sl1 = bmtl1_sl1_buffer.size();
      LogDebug("ShowerBuilder") << "        o Shower found in SL1 with " << nHits_sl1 << " hits";
    }
    showerCands[1]->flag();
    set_shower_properties(showerCands[1], bmtl1_sl1_buffer);
  }
  if (triggerShower(bmtl1_sl3_buffer)) {
    if (debug_) {
      int nHits_sl3 = bmtl1_sl3_buffer.size();
      LogDebug("ShowerBuilder") << "        o Shower found in SL3 with " << nHits_sl3 << " hits";
    }
    showerCands[3]->flag();
    set_shower_properties(showerCands[3], bmtl1_sl3_buffer);
  }
}

void ShowerBuilder::processHitsFirmwareEmulation(std::map<int, ShowerCandidatePtr> &showerCands) {
  // Use a for over BXs to emulate the behavior of the OBDT and BMTL1, this means considering:
  // 1. OBDT can only store recived hits during 4 BXs (adjustable with obdt_hits_bxpersistence_)
  // 2. OBDT can recive hits from the same wire during 2 BXs (adjustable with obdt_wire_relaxing_time_)
  // 3. OBDT can only sends 8 hits per BX to the BMTL1 (adjustable with obdt_hits_per_bx_)
  // 4. Shower algorithm in the BMTL1 mantains the hits along 16 BXs and triggers if nHits >= threshold (adjustable with bmtl1_hits_bxpersistence_)
  groupHits_byBx();

  // auxiliary variables
  int prev_nHits_sl1 = 0;
  int nHits_sl1 = 0;
  bool shower_sl1_already_set = false;
  int prev_nHits_sl3 = 0;
  int nHits_sl3 = 0;
  bool shower_sl3_already_set = false;
  // -------------------

  int min_bx = all_hits_perBx.begin()->first;
  int max_bx = all_hits_perBx.rbegin()->first;

  if (debug_)
    LogDebug("ShowerBuilder") << "        - bx range: " << min_bx << " - " << max_bx << ", size: " << (max_bx - min_bx);

  for (int bx = min_bx; bx <= max_bx + 17; bx++) {
    fill_obdt(bx);

    if (debug_)
      LogDebug("ShowerBuilder") << "          ^ " << obdt_buffer.size() << " hits in obdt_buffer at BX " << bx;
    fill_bmtl1_buffers();

    nHits_sl1 = bmtl1_sl1_buffer.size();
    if (debug_)
      LogDebug("ShowerBuilder") << "          ^ " << nHits_sl1 << " hits in bmtl1_sl1_buffer at BX " << bx;
    nHits_sl3 = bmtl1_sl3_buffer.size();
    if (debug_)
      LogDebug("ShowerBuilder") << "          ^ " << nHits_sl3 << " hits in bmtl1_sl3_buffer at BX " << bx;
    if (triggerShower(bmtl1_sl1_buffer)) {
      if (debug_ && !showerCands[1]->isFlagged()) {
        LogDebug("ShowerBuilder") << "        o Shower found in SL1 with " << nHits_sl1 << " hits";
      }
      showerCands[1]->flag();
    }
    if (triggerShower(bmtl1_sl3_buffer)) {
      if (debug_ && !showerCands[3]->isFlagged()) {
        LogDebug("ShowerBuilder") << "        o Shower found in SL3 with " << nHits_sl3 << " hits";
      }
      showerCands[3]->flag();
    }

    if (nHits_sl1 < prev_nHits_sl1 && showerCands[1]->isFlagged() && !shower_sl1_already_set) {
      shower_sl1_already_set = true;
      set_shower_properties(showerCands[1], bmtl1_sl1_buffer, nHits_sl1, bx);
    } else {
      prev_nHits_sl1 = nHits_sl1;
    }

    if (nHits_sl3 < prev_nHits_sl3 && showerCands[3]->isFlagged() && !shower_sl3_already_set) {
      shower_sl3_already_set = true;
      set_shower_properties(showerCands[3], bmtl1_sl3_buffer, nHits_sl3, bx);
    } else {
      prev_nHits_sl3 = nHits_sl3;
    }

    bxStep(bx);
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
                                          int bx,
                                          int min_wire,
                                          int max_wire,
                                          float avg_pos,
                                          float avg_time) {
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
  showerCand->setBX(bx);
  showerCand->setMinWire(min_wire);
  showerCand->setMaxWire(max_wire);
  showerCand->setAvgPos(avg_pos);
  showerCand->setAvgTime(avg_time);
  showerb::set_wires_profile(showerCand->getWiresProfile(), _hits);
}

void ShowerBuilder::groupHits_byBx() {
  double shift_back = 0;
  if (scenario_ == MC)  //scope for MC
    shift_back = 400;
  else if (scenario_ == DATA)  //scope for data
    shift_back = 0;
  else if (scenario_ == SLICE_TEST)  //scope for slice test
    shift_back = 400;

  all_hits_perBx.clear();
  // Group hits by BX
  for (auto &hit : all_hits) {
    // Compute the BX from the TDC time
    int bx = hit.tdcTimeStamp() / 25 - shift_back;
    all_hits_perBx[bx].push_back(hit);
  }
}

void ShowerBuilder::fill_obdt(const int bx) {
  // Fill the OBDT buffer with the hits in the current BX this function ensure that hot wires are not added
  if (all_hits_perBx.find(bx) != all_hits_perBx.end()) {
    for (auto &hit : all_hits_perBx[bx]) {
      if (debug_)
        LogDebug("ShowerBuilder") << "          ^ Trying to add hit with wire " << hit.channelId() << " in OBDT at BX "
                                  << bx;
      if (!showerb::buffer_contains(hot_wires_buffer, hit)) {
        if (debug_)
          LogDebug("ShowerBuilder") << "          ^ added";
        showerb::DTPrimPlusBx _hit(bx, hit);
        obdt_buffer.push_back(_hit);
        hot_wires_buffer.push_back(_hit);
      }
    }
  }
}

void ShowerBuilder::fill_bmtl1_buffers() {
  // Fill the BMTL1 buffer with the hits in the OBDT buffer only nHits_per_bx_ hits are added
  if (obdt_buffer.empty())
    return;
  if (debug_)
    LogDebug("ShowerBuilder") << "          ^ Getting hits from OBDT";
  for (int i = 0; i < nHits_per_bx_; i++) {
    if (obdt_buffer.empty())
      break;
    auto _hitpbx = obdt_buffer.front();
    if (_hitpbx.second.superLayerId() == 1) {
      bmtl1_sl1_buffer.push_back(_hitpbx);
    } else if (_hitpbx.second.superLayerId() == 3) {
      bmtl1_sl3_buffer.push_back(_hitpbx);
    }
    obdt_buffer.pop_front();
  }
}

void ShowerBuilder::bxStep(const int _current_bx) {
  // Remove old elements from the buffers
  showerb::buffer_clear_olds(obdt_buffer, _current_bx, obdt_hits_bxpersistence_);
  showerb::buffer_clear_olds(hot_wires_buffer, _current_bx, obdt_hits_bxpersistence_);
  showerb::buffer_clear_olds(bmtl1_sl1_buffer, _current_bx, bmtl1_hits_bxpersistence_);
  showerb::buffer_clear_olds(bmtl1_sl3_buffer, _current_bx, bmtl1_hits_bxpersistence_);
}
