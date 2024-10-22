#include "L1Trigger/DTTriggerPhase2/interface/MuonPathConfirmator.h"
#include <cmath>
#include <memory>

using namespace edm;
using namespace std;
using namespace cmsdt;

// ============================================================================
// Constructors and destructor
// ============================================================================
MuonPathConfirmator::MuonPathConfirmator(const ParameterSet &pset, edm::ConsumesCollector &iC)
    : debug_(pset.getUntrackedParameter<bool>("debug")),
      minx_match_2digis_(pset.getParameter<double>("minx_match_2digis")) {
  if (debug_)
    LogDebug("MuonPathConfirmator") << "MuonPathConfirmator: constructor";

  //shift phi
  int rawId;
  shift_filename_ = pset.getParameter<edm::FileInPath>("shift_filename");
  std::ifstream ifin3(shift_filename_.fullPath());
  double shift;
  if (ifin3.fail()) {
    throw cms::Exception("Missing Input File")
        << "MuonPathConfirmator::MuonPathConfirmator() -  Cannot find " << shift_filename_.fullPath();
  }
  while (ifin3.good()) {
    ifin3 >> rawId >> shift;
    shiftinfo_[rawId] = shift;
  }

  int wh, st, se, maxdrift;
  maxdrift_filename_ = pset.getParameter<edm::FileInPath>("maxdrift_filename");
  std::ifstream ifind(maxdrift_filename_.fullPath());
  if (ifind.fail()) {
    throw cms::Exception("Missing Input File")
        << "MPSLFilter::MPSLFilter() -  Cannot find " << maxdrift_filename_.fullPath();
  }
  while (ifind.good()) {
    ifind >> wh >> st >> se >> maxdrift;
    maxdriftinfo_[wh][st][se] = maxdrift;
  }
}

MuonPathConfirmator::~MuonPathConfirmator() {
  if (debug_)
    LogDebug("MuonPathConfirmator") << "MuonPathAnalyzer: destructor";
}

// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================

void MuonPathConfirmator::run(edm::Event &iEvent,
                              const edm::EventSetup &iEventSetup,
                              std::vector<cmsdt::metaPrimitive> inMetaPrimitives,
                              edm::Handle<DTDigiCollection> dtdigis,
                              std::vector<cmsdt::metaPrimitive> &outMetaPrimitives) {
  if (debug_)
    LogDebug("MuonPathConfirmator") << "MuonPathConfirmator: run";

  // fit per SL (need to allow for multiple outputs for a single mpath)
  if (!inMetaPrimitives.empty()) {
    int dum_sl_rawid = inMetaPrimitives[0].rawId;
    DTSuperLayerId dumSlId(dum_sl_rawid);
    DTChamberId ChId(dumSlId.wheel(), dumSlId.station(), dumSlId.sector());
    max_drift_tdc = maxdriftinfo_[dumSlId.wheel() + 2][dumSlId.station() - 1][dumSlId.sector() - 1];
  }

  for (auto &mp : inMetaPrimitives) {
    analyze(mp, dtdigis, outMetaPrimitives);
  }
}

void MuonPathConfirmator::initialise(const edm::EventSetup &iEventSetup) {
  if (debug_)
    LogDebug("MuonPathConfirmator") << "MuonPathConfirmator::initialiase";
}

void MuonPathConfirmator::finish() {
  if (debug_)
    LogDebug("MuonPathConfirmator") << "MuonPathConfirmator: finish";
};

//------------------------------------------------------------------
//--- Metodos privados
//------------------------------------------------------------------

void MuonPathConfirmator::analyze(cmsdt::metaPrimitive mp,
                                  edm::Handle<DTDigiCollection> dtdigis,
                                  std::vector<cmsdt::metaPrimitive> &outMetaPrimitives) {
  int dum_sl_rawid = mp.rawId;
  DTSuperLayerId dumSlId(dum_sl_rawid);
  DTChamberId ChId(dumSlId.wheel(), dumSlId.station(), dumSlId.sector());
  DTSuperLayerId sl1Id(ChId.rawId(), 1);
  DTSuperLayerId sl3Id(ChId.rawId(), 3);

  DTWireId wireIdSL1(sl1Id, 2, 1);
  DTWireId wireIdSL3(sl3Id, 2, 1);
  auto sl_shift_cm = shiftinfo_[wireIdSL1.rawId()] - shiftinfo_[wireIdSL3.rawId()];
  bool isSL1 = (mp.rawId == sl1Id.rawId());
  bool isSL3 = (mp.rawId == sl3Id.rawId());
  if (!isSL1 && !isSL3)
    outMetaPrimitives.emplace_back(mp);
  else {
    int best_tdc = -1;
    int next_tdc = -1;
    int best_wire = -1;
    int next_wire = -1;
    int best_layer = -1;
    int next_layer = -1;
    int best_lat = -1;
    int next_lat = -1;
    int lat = -1;
    int matched_digis = 0;

    int position_prec = ((int)(mp.x)) << PARTIALS_PRECISSION;
    int slope_prec = ((int)(mp.tanPhi)) << PARTIALS_PRECISSION;

    int slope_x_halfchamb = (((long int)slope_prec) * SEMICHAMBER_H) >> SEMICHAMBER_RES_SHR;
    int slope_x_3semicells = (slope_prec * 3) >> LYRANDAHALF_RES_SHR;
    int slope_x_1semicell = (slope_prec * 1) >> LYRANDAHALF_RES_SHR;

    for (const auto &dtLayerId_It : *dtdigis) {
      const DTLayerId dtLId = dtLayerId_It.first;
      // creating a new DTSuperLayerId object to compare with the required SL id
      const DTSuperLayerId dtSLId(dtLId.wheel(), dtLId.station(), dtLId.sector(), dtLId.superLayer());
      bool hitFromSL1 = (dtSLId.rawId() == sl1Id.rawId());
      bool hitFromSL3 = (dtSLId.rawId() == sl3Id.rawId());
      if (!(hitFromSL1 || hitFromSL3))  // checking hits are from one of the other SL of the same chamber
        continue;
      double minx = 10 * minx_match_2digis_ * ((double)max_drift_tdc / (double)CELL_SEMILENGTH);
      double min2x = 10 * minx_match_2digis_ * ((double)max_drift_tdc / (double)CELL_SEMILENGTH);
      if (isSL1 != hitFromSL1) {  // checking hits have the opposite SL than the TP
        for (auto digiIt = (dtLayerId_It.second).first; digiIt != (dtLayerId_It.second).second; ++digiIt) {
          if ((*digiIt).time() < mp.t0)
            continue;
          int wp_semicells = ((*digiIt).wire() - 1 - SL1_CELLS_OFFSET) * 2 + 1;
          int ly = dtLId.layer() - 1;
          if (ly % 2 == 1)
            wp_semicells -= 1;
          if (hitFromSL3)
            wp_semicells -= (int)round((sl_shift_cm * 10) / CELL_SEMILENGTH);
          double hit_position = wp_semicells * max_drift_tdc +
                                ((*digiIt).time() - mp.t0) * (double)TIME_TO_TDC_COUNTS / (double)LHC_CLK_FREQ;
          double hit_position_left = wp_semicells * max_drift_tdc -
                                     ((*digiIt).time() - mp.t0) * (double)TIME_TO_TDC_COUNTS / (double)LHC_CLK_FREQ;
          // extrapolating position to the layer of the hit
          // mp.position is referred to the center between SLs, so one has to add half the distance between SLs
          // + half a cell height to get to the first wire + ly * cell height to reach the desired ly
          // 10 * VERT_PHI1_PHI3 / 2 + (CELL_HEIGHT / 2) + ly * CELL_HEIGHT = (10 * VERT_PHI1_PHI3 + (2 * ly + 1) * CELL_HEIGHT) / 2

          int position_in_layer = position_prec + (1 - 2 * (int)hitFromSL1) * slope_x_halfchamb;
          if (ly == 0)
            position_in_layer -= slope_x_3semicells;
          if (ly == 1)
            position_in_layer -= slope_x_1semicell;
          if (ly == 2)
            position_in_layer += slope_x_1semicell;
          if (ly == 3)
            position_in_layer += slope_x_3semicells;
          position_in_layer = position_in_layer >> PARTIALS_PRECISSION;

          if (std::abs(position_in_layer - hit_position_left) < std::abs(position_in_layer - hit_position)) {
            lat = 0;
            hit_position = hit_position_left;
          }
          if (std::abs(position_in_layer - hit_position) < minx) {
            // different layer than the stored in best, hit added, matched_digis++;. This approach in somewhat
            // buggy, as we could have stored as best LayerX -> LayerY -> LayerX, and this should
            // count only as 2 hits. However, as we confirm with at least 2 hits, having 2 or more
            // makes no difference
            if (dtLId.layer() != best_layer) {
              minx = std::abs(position_in_layer - hit_position);
              next_wire = best_wire;
              next_tdc = best_tdc;
              next_layer = best_layer;
              next_lat = best_lat;
              matched_digis++;
            }
            best_wire = (*digiIt).wire() - 1;
            best_tdc = (*digiIt).time();
            best_layer = dtLId.layer();
            best_lat = lat;

          } else if ((std::abs(position_in_layer - hit_position) >= minx) &&
                     (std::abs(position_in_layer - hit_position) < min2x)) {
            // same layer than the stored in best, no hit added
            if (dtLId.layer() == best_layer)
              continue;
            // different layer than the stored in next, hit added. This approach in somewhat
            // buggy, as we could have stored as next LayerX -> LayerY -> LayerX, and this should
            // count only as 2 hits. However, as we confirm with at least 2 hits, having 2 or more
            // makes no difference
            matched_digis++;
            // whether the layer is the same for this hit and the stored in next, we substitute
            // the one stored and modify the min distance
            min2x = std::abs(position_in_layer - hit_position);
            next_wire = (*digiIt).wire() - 1;
            next_tdc = (*digiIt).time();
            next_layer = dtLId.layer();
            next_lat = lat;
          }
        }
      }
    }
    int new_quality = mp.quality;
    std::vector<int> wi_c(4, -1), tdc_c(4, -1), lat_c(4, -1);
    if (matched_digis >= 2 and best_layer != -1 and next_layer != -1) {  // actually confirm
      new_quality = CHIGHQ;
      if (mp.quality == LOWQ)
        new_quality = CLOWQ;

      wi_c[next_layer - 1] = next_wire;
      tdc_c[next_layer - 1] = next_tdc;
      lat_c[next_layer - 1] = next_lat;

      wi_c[best_layer - 1] = best_wire;
      tdc_c[best_layer - 1] = best_tdc;
      lat_c[best_layer - 1] = best_lat;
    }
    if (isSL1) {
      outMetaPrimitives.emplace_back(metaPrimitive(
          {mp.rawId, mp.t0,   mp.x,     mp.tanPhi, mp.phi,   mp.phiB, mp.phi_cmssw, mp.phiB_cmssw, mp.chi2, new_quality,
           mp.wi1,   mp.tdc1, mp.lat1,  mp.wi2,    mp.tdc2,  mp.lat2, mp.wi3,       mp.tdc3,       mp.lat3, mp.wi4,
           mp.tdc4,  mp.lat4, wi_c[0],  tdc_c[0],  lat_c[0], wi_c[1], tdc_c[1],     lat_c[1],      wi_c[2], tdc_c[2],
           lat_c[2], wi_c[3], tdc_c[3], lat_c[3],  -1}));
    } else {
      outMetaPrimitives.emplace_back(
          metaPrimitive({mp.rawId,      mp.t0,    mp.x,        mp.tanPhi, mp.phi,   mp.phiB,  mp.phi_cmssw,
                         mp.phiB_cmssw, mp.chi2,  new_quality, wi_c[0],   tdc_c[0], lat_c[0], wi_c[1],
                         tdc_c[1],      lat_c[1], wi_c[2],     tdc_c[2],  lat_c[2], wi_c[3],  tdc_c[3],
                         lat_c[3],      mp.wi5,   mp.tdc5,     mp.lat5,   mp.wi6,   mp.tdc6,  mp.lat6,
                         mp.wi7,        mp.tdc7,  mp.lat7,     mp.wi8,    mp.tdc8,  mp.lat8,  -1}));
    }
  }  //SL2
}
