#include "L1Trigger/DTTriggerPhase2/interface/MuonPathSLFitter.h"
#include <cmath>
#include <memory>

using namespace edm;
using namespace std;
using namespace cmsdt;
// ============================================================================
// Constructors and destructor
// ============================================================================
MuonPathSLFitter::MuonPathSLFitter(const ParameterSet &pset,
                                   edm::ConsumesCollector &iC,
                                   std::shared_ptr<GlobalCoordsObtainer> &globalcoordsobtainer)
    : MuonPathFitter(pset, iC, globalcoordsobtainer) {
  if (debug_)
    LogDebug("MuonPathSLFitter") << "MuonPathSLFitter: constructor";

  //shift theta
  int rawId;
  double shift;
  shift_theta_filename_ = pset.getParameter<edm::FileInPath>("shift_theta_filename");
  std::ifstream ifin4(shift_theta_filename_.fullPath());
  if (ifin4.fail()) {
    throw cms::Exception("Missing Input File")
        << "MuonPathSLFitter::MuonPathSLFitter() -  Cannot find " << shift_theta_filename_.fullPath();
  }

  while (ifin4.good()) {
    ifin4 >> rawId >> shift;
    shiftthetainfo_[rawId] = shift;
  }

  // LUTs
  sl1_filename_ = pset.getParameter<edm::FileInPath>("lut_sl1");
  sl2_filename_ = pset.getParameter<edm::FileInPath>("lut_sl2");
  sl3_filename_ = pset.getParameter<edm::FileInPath>("lut_sl3");

  fillLuts();

  setChi2Th(pset.getParameter<double>("chi2Th"));
  setTanPhiTh(pset.getParameter<double>("tanPhiTh"));
}

MuonPathSLFitter::~MuonPathSLFitter() {
  if (debug_)
    LogDebug("MuonPathSLFitter") << "MuonPathSLFitter: destructor";
}

// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void MuonPathSLFitter::initialise(const edm::EventSetup &iEventSetup) {
  if (debug_)
    LogDebug("MuonPathSLFitter") << "MuonPathSLFitter::initialiase";

  auto geom = iEventSetup.getHandle(dtGeomH);
  dtGeo_ = &(*geom);
}

void MuonPathSLFitter::run(edm::Event &iEvent,
                           const edm::EventSetup &iEventSetup,
                           MuonPathPtrs &muonpaths,
                           std::vector<lat_vector> &lateralities,
                           std::vector<metaPrimitive> &metaPrimitives) {
  if (debug_)
    LogDebug("MuonPathSLFitter") << "MuonPathSLFitter: run";

  // fit per SL (need to allow for multiple outputs for a single mpath)
  // for (auto &muonpath : muonpaths) {
  if (!muonpaths.empty()) {
    auto muonpath = muonpaths[0];
    int rawId = muonpath->primitive(0)->cameraId();
    if (muonpath->primitive(0)->cameraId() == -1) {
      rawId = muonpath->primitive(1)->cameraId();
    }
    const DTLayerId dtLId(rawId);
    max_drift_tdc = maxdriftinfo_[dtLId.wheel() + 2][dtLId.station() - 1][dtLId.sector() - 1];
  }

  for (size_t i = 0; i < muonpaths.size(); i++) {
    auto muonpath = muonpaths[i];
    auto lats = lateralities[i];
    analyze(muonpath, lats, metaPrimitives);
  }
}

void MuonPathSLFitter::finish() {
  if (debug_)
    LogDebug("MuonPathSLFitter") << "MuonPathSLFitter: finish";
};

//------------------------------------------------------------------
//--- Metodos privados
//------------------------------------------------------------------

void MuonPathSLFitter::analyze(MuonPathPtr &inMPath,
                               lat_vector lat_combs,
                               std::vector<cmsdt::metaPrimitive> &metaPrimitives) {
  auto sl = inMPath->primitive(0)->superLayerId();  // 0, 1, 2

  int selected_lay = 1;
  if (inMPath->primitive(0)->tdcTimeStamp() != -1)
    selected_lay = 0;

  int dumLayId = inMPath->primitive(selected_lay)->cameraId();
  auto dtDumlayerId = DTLayerId(dumLayId);
  DTSuperLayerId MuonPathSLId(dtDumlayerId.wheel(), dtDumlayerId.station(), dtDumlayerId.sector(), sl + 1);

  DTChamberId ChId(MuonPathSLId.wheel(), MuonPathSLId.station(), MuonPathSLId.sector());

  DTSuperLayerId MuonPathSL1Id(dtDumlayerId.wheel(), dtDumlayerId.station(), dtDumlayerId.sector(), 1);
  DTSuperLayerId MuonPathSL2Id(dtDumlayerId.wheel(), dtDumlayerId.station(), dtDumlayerId.sector(), 2);
  DTSuperLayerId MuonPathSL3Id(dtDumlayerId.wheel(), dtDumlayerId.station(), dtDumlayerId.sector(), 3);
  DTWireId wireIdSL1(MuonPathSL1Id, 2, 1);
  DTWireId wireIdSL2(MuonPathSL2Id, 2, 1);
  DTWireId wireIdSL3(MuonPathSL3Id, 2, 1);
  auto sl_shift_cm = shiftinfo_[wireIdSL1.rawId()] - shiftinfo_[wireIdSL3.rawId()];

  fit_common_in_t fit_common_in;

  // 8-element vectors, for the 8 layers. As here we are fitting one SL only, we leave the other SL values as dummy ones
  fit_common_in.hits = {};
  fit_common_in.hits_valid = {};

  int quality = 3;
  if (inMPath->missingLayer() != -1)
    quality = 1;

  int minISL = 1;
  int maxISL = 3;
  if (sl < 1) {
    minISL = 0;
    maxISL = 2;
  }

  for (int isl = minISL; isl < maxISL; isl++) {
    for (int i = 0; i < NUM_LAYERS; i++) {
      if (isl == sl && inMPath->missingLayer() != i) {
        // Include both valid and non-valid hits. Non-valid values can be whatever, leaving all as -1 to make debugging easier.
        auto ti = inMPath->primitive(i)->tdcTimeStamp();
        if (ti != -1)
          ti = (int)round(((float)TIME_TO_TDC_COUNTS / (float)LHC_CLK_FREQ) * ti);
        auto wi = inMPath->primitive(i)->channelId();
        auto ly = inMPath->primitive(i)->layerId();
        // int layId = inMPath->primitive(i)->cameraId();
        // auto dtlayerId = DTLayerId(layId);
        // auto wireId = DTWireId(dtlayerId, wi + 1); // wire start from 1, mixer groups them starting from 0
        // int rawId = wireId.rawId();
        // wp in tdc counts (still in floating point)
        int wp_semicells = (wi - SL1_CELLS_OFFSET) * 2 + 1;
        if (ly % 2 == 1)
          wp_semicells -= 1;
        if (isl == 2)
          wp_semicells -= (int)round((sl_shift_cm * 10) / CELL_SEMILENGTH);

        float wp_tdc = wp_semicells * max_drift_tdc;
        int wp = (int)((long int)(round(wp_tdc * std::pow(2, WIREPOS_WIDTH))) / (int)std::pow(2, WIREPOS_WIDTH));
        fit_common_in.hits.push_back({ti, wi, ly, wp});
        // fill valids as well
        if (inMPath->missingLayer() == i)
          fit_common_in.hits_valid.push_back(0);
        else
          fit_common_in.hits_valid.push_back(1);
      } else {
        fit_common_in.hits.push_back({-1, -1, -1, -1});
        fit_common_in.hits_valid.push_back(0);
      }
    }
  }

  int smallest_time = 999999, tmp_coarse_wirepos_1 = -1, tmp_coarse_wirepos_3 = -1;
  // coarse_bctr is the 12 MSB of the smallest tdc
  for (int isl = 0; isl < 3; isl++) {
    if (isl != sl)
      continue;
    int myisl = isl < 2 ? 0 : 1;
    for (size_t i = 0; i < NUM_LAYERS; i++) {
      if (fit_common_in.hits_valid[NUM_LAYERS * myisl + i] == 0)
        continue;
      else if (fit_common_in.hits[NUM_LAYERS * myisl + i].ti < smallest_time)
        smallest_time = fit_common_in.hits[NUM_LAYERS * myisl + i].ti;
    }
    if (fit_common_in.hits_valid[NUM_LAYERS * myisl + 0] == 1)
      tmp_coarse_wirepos_1 = fit_common_in.hits[NUM_LAYERS * myisl + 0].wp;
    else
      tmp_coarse_wirepos_1 = fit_common_in.hits[NUM_LAYERS * myisl + 1].wp;
    if (fit_common_in.hits_valid[NUM_LAYERS * myisl + 3] == 1)
      tmp_coarse_wirepos_3 = fit_common_in.hits[NUM_LAYERS * myisl + 3].wp;
    else
      tmp_coarse_wirepos_3 = fit_common_in.hits[NUM_LAYERS * myisl + 2].wp;

    tmp_coarse_wirepos_1 = tmp_coarse_wirepos_1 >> WIREPOS_NORM_LSB_IGNORED;
    tmp_coarse_wirepos_3 = tmp_coarse_wirepos_3 >> WIREPOS_NORM_LSB_IGNORED;
  }
  fit_common_in.coarse_bctr = smallest_time >> (WIDTH_FULL_TIME - WIDTH_COARSED_TIME);
  fit_common_in.coarse_wirepos = (tmp_coarse_wirepos_1 + tmp_coarse_wirepos_3) >> 1;

  for (auto &lat_comb : lat_combs) {
    if (lat_comb[0] == 0 && lat_comb[1] == 0 && lat_comb[2] == 0 && lat_comb[3] == 0)
      continue;
    fit_common_in.lateralities.clear();

    auto rom_addr = get_rom_addr(inMPath, lat_comb);
    coeffs_t coeffs;
    if (sl == 0) {
      coeffs =
          RomDataConvert(lut_sl1[rom_addr], COEFF_WIDTH_SL_T0, COEFF_WIDTH_SL_POSITION, COEFF_WIDTH_SL_SLOPE, 0, 3);
    } else if (sl == 1) {
      coeffs =
          RomDataConvert(lut_sl2[rom_addr], COEFF_WIDTH_SL_T0, COEFF_WIDTH_SL2_POSITION, COEFF_WIDTH_SL_SLOPE, 0, 3);
    } else {
      coeffs =
          RomDataConvert(lut_sl3[rom_addr], COEFF_WIDTH_SL_T0, COEFF_WIDTH_SL_POSITION, COEFF_WIDTH_SL_SLOPE, 4, 7);
    }
    // Filling lateralities
    int minISL = 1;
    int maxISL = 3;
    if (sl < 1) {
      minISL = 0;
      maxISL = 2;
    }

    for (int isl = minISL; isl < maxISL; isl++) {
      for (size_t i = 0; i < NUM_LAYERS; i++) {
        if (isl == sl) {
          fit_common_in.lateralities.push_back(lat_comb[i]);
        } else
          fit_common_in.lateralities.push_back(-1);
      }
    }
    fit_common_in.coeffs = coeffs;

    auto fit_common_out = fit(fit_common_in,
                              XI_SL_WIDTH,
                              COEFF_WIDTH_SL_T0,
                              sl == 1 ? COEFF_WIDTH_SL2_POSITION : COEFF_WIDTH_SL_POSITION,
                              COEFF_WIDTH_SL_SLOPE,
                              PRECISSION_SL_T0,
                              PRECISSION_SL_POSITION,
                              PRECISSION_SL_SLOPE,
                              PROD_RESIZE_SL_T0,
                              sl == 1 ? PROD_RESIZE_SL2_POSITION : PROD_RESIZE_SL_POSITION,
                              PROD_RESIZE_SL_SLOPE,
                              max_drift_tdc,
                              sl + 1);

    if (fit_common_out.valid_fit == 1) {
      float t0_f = ((float)fit_common_out.t0) * (float)LHC_CLK_FREQ / (float)TIME_TO_TDC_COUNTS;

      float slope_f = -fit_common_out.slope * ((float)CELL_SEMILENGTH / max_drift_tdc) * (1) / (CELL_SEMIHEIGHT * 16.);
      if (sl != 1 && std::abs(slope_f) > tanPhiTh_)
        continue;

      DTWireId wireId(MuonPathSLId, 2, 1);
      float pos_ch_f = (float)(fit_common_out.position) * ((float)CELL_SEMILENGTH / (float)max_drift_tdc) / 10;
      pos_ch_f += (SL1_CELLS_OFFSET * CELL_LENGTH) / 10.;
      if (sl != 1)
        pos_ch_f += shiftinfo_[wireIdSL1.rawId()];
      else if (sl == 1)
        pos_ch_f += shiftthetainfo_[wireIdSL2.rawId()];

      float pos_sl_f = pos_ch_f - (sl - 1) * slope_f * VERT_PHI1_PHI3 / 2;
      float chi2_f = fit_common_out.chi2 * std::pow(((float)CELL_SEMILENGTH / (float)max_drift_tdc), 2) / 100;

      // obtention of global coordinates using luts
      int pos = (int)(10 * (pos_sl_f - shiftinfo_[wireId.rawId()]) * INCREASED_RES_POS_POW);
      int slope = (int)(-slope_f * INCREASED_RES_SLOPE_POW);
      auto global_coords = globalcoordsobtainer_->get_global_coordinates(ChId.rawId(), sl + 1, pos, slope);
      float phi = global_coords[0];
      float phiB = global_coords[1];

      // obtention of global coordinates using cmssw geometry
      double z = 0;
      if (ChId.station() == 3 or ChId.station() == 4) {
        z = Z_SHIFT_MB4;
      }
      GlobalPoint jm_x_cmssw_global = dtGeo_->chamber(ChId)->toGlobal(LocalPoint(pos_sl_f, 0., z));
      int thisec = ChId.sector();
      if (thisec == 13)
        thisec = 4;
      if (thisec == 14)
        thisec = 10;
      float phi_cmssw = jm_x_cmssw_global.phi() - PHI_CONV * (thisec - 1);
      float psi = atan(slope_f);
      float phiB_cmssw = hasPosRF(ChId.wheel(), ChId.sector()) ? psi - phi_cmssw : -psi - phi_cmssw;
      if (sl == 0)
        metaPrimitives.emplace_back(metaPrimitive({MuonPathSLId.rawId(),
                                                   t0_f,
                                                   (double)(fit_common_out.position),
                                                   (double)fit_common_out.slope,
                                                   phi,
                                                   phiB,
                                                   phi_cmssw,
                                                   phiB_cmssw,
                                                   chi2_f,
                                                   quality,
                                                   inMPath->primitive(0)->channelId(),
                                                   inMPath->primitive(0)->tdcTimeStamp(),
                                                   lat_comb[0],
                                                   inMPath->primitive(1)->channelId(),
                                                   inMPath->primitive(1)->tdcTimeStamp(),
                                                   lat_comb[1],
                                                   inMPath->primitive(2)->channelId(),
                                                   inMPath->primitive(2)->tdcTimeStamp(),
                                                   lat_comb[2],
                                                   inMPath->primitive(3)->channelId(),
                                                   inMPath->primitive(3)->tdcTimeStamp(),
                                                   lat_comb[3],
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1}));
      else if (sl == 2)
        metaPrimitives.emplace_back(metaPrimitive({MuonPathSLId.rawId(),
                                                   t0_f,
                                                   (double)(fit_common_out.position),
                                                   (double)fit_common_out.slope,
                                                   phi,
                                                   phiB,
                                                   phi_cmssw,
                                                   phiB_cmssw,
                                                   chi2_f,
                                                   quality,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   inMPath->primitive(0)->channelId(),
                                                   inMPath->primitive(0)->tdcTimeStamp(),
                                                   lat_comb[0],
                                                   inMPath->primitive(1)->channelId(),
                                                   inMPath->primitive(1)->tdcTimeStamp(),
                                                   lat_comb[1],
                                                   inMPath->primitive(2)->channelId(),
                                                   inMPath->primitive(2)->tdcTimeStamp(),
                                                   lat_comb[2],
                                                   inMPath->primitive(3)->channelId(),
                                                   inMPath->primitive(3)->tdcTimeStamp(),
                                                   lat_comb[3],
                                                   -1}));
      else if (sl == 1) {
        // fw-like calculation
        DTLayerId SL2_layer2Id(MuonPathSLId, 2);
        double z_shift = shiftthetainfo_[SL2_layer2Id.rawId()];
        double jm_y = hasPosRF(MuonPathSLId.wheel(), MuonPathSLId.sector()) ? z_shift - pos : z_shift + pos;
        phi = jm_y;
        phiB = slope_f;

        // cmssw-like calculation
        LocalPoint wire1_in_layer(dtGeo_->layer(SL2_layer2Id)->specificTopology().wirePosition(1), 0, -0.65);
        GlobalPoint wire1_in_global = dtGeo_->layer(SL2_layer2Id)->toGlobal(wire1_in_layer);
        LocalPoint wire1_in_sl = dtGeo_->superLayer(MuonPathSLId)->toLocal(wire1_in_global);
        double x_shift = wire1_in_sl.x();
        jm_y = (dtGeo_->superLayer(MuonPathSLId)
                    ->toGlobal(LocalPoint(double(pos) / (10 * pow(2, INCREASED_RES_POS)) + x_shift, 0., 0)))
                   .z();
        phi_cmssw = jm_y;
        phiB_cmssw = slope_f;
        metaPrimitives.emplace_back(metaPrimitive({MuonPathSLId.rawId(),
                                                   t0_f,
                                                   (double)(fit_common_out.position),
                                                   (double)fit_common_out.slope,
                                                   phi,
                                                   phiB,
                                                   phi_cmssw,
                                                   phiB_cmssw,
                                                   chi2_f,
                                                   quality,
                                                   inMPath->primitive(0)->channelId(),
                                                   inMPath->primitive(0)->tdcTimeStamp(),
                                                   lat_comb[0],
                                                   inMPath->primitive(1)->channelId(),
                                                   inMPath->primitive(1)->tdcTimeStamp(),
                                                   lat_comb[1],
                                                   inMPath->primitive(2)->channelId(),
                                                   inMPath->primitive(2)->tdcTimeStamp(),
                                                   lat_comb[2],
                                                   inMPath->primitive(3)->channelId(),
                                                   inMPath->primitive(3)->tdcTimeStamp(),
                                                   lat_comb[3],
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1,
                                                   -1}));
      }
    }  // (fit_common_out.valid_fit == 1)
  }    // loop in lat_combs
  return;
}

void MuonPathSLFitter::fillLuts() {
  std::ifstream ifinsl1(sl1_filename_.fullPath());
  std::string line;
  while (ifinsl1.good()) {
    ifinsl1 >> line;

    std::vector<int> myNumbers;
    for (size_t i = 0; i < line.size(); i++) {
      // This converts the char into an int and pushes it into vec
      myNumbers.push_back(line[i] - '0');  // The digits will be in the same order as before
    }
    std::reverse(myNumbers.begin(), myNumbers.end());
    lut_sl1.push_back(myNumbers);
  }

  std::ifstream ifinsl2(sl2_filename_.fullPath());
  while (ifinsl2.good()) {
    ifinsl2 >> line;

    std::vector<int> myNumbers;
    for (size_t i = 0; i < line.size(); i++) {
      // This converts the char into an int and pushes it into vec
      myNumbers.push_back(line[i] - '0');  // The digits will be in the same order as before
    }
    std::reverse(myNumbers.begin(), myNumbers.end());
    lut_sl2.push_back(myNumbers);
  }

  std::ifstream ifinsl3(sl3_filename_.fullPath());
  while (ifinsl3.good()) {
    ifinsl3 >> line;

    std::vector<int> myNumbers;
    for (size_t i = 0; i < line.size(); i++) {
      // This converts the char into an int and pushes it into vec
      myNumbers.push_back(line[i] - '0');  // The digits will be in the same order as before
    }
    std::reverse(myNumbers.begin(), myNumbers.end());
    lut_sl3.push_back(myNumbers);
  }

  return;
}

int MuonPathSLFitter::get_rom_addr(MuonPathPtr &inMPath, latcomb lats) {
  std::vector<int> rom_addr;
  auto missing_layer = inMPath->missingLayer();
  if (missing_layer == -1) {
    rom_addr.push_back(1);
    rom_addr.push_back(0);
  } else {
    if (missing_layer == 0) {
      rom_addr.push_back(0);
      rom_addr.push_back(0);
    } else if (missing_layer == 1) {
      rom_addr.push_back(0);
      rom_addr.push_back(1);
    } else if (missing_layer == 2) {
      rom_addr.push_back(1);
      rom_addr.push_back(0);
    } else {  // missing_layer == 3
      rom_addr.push_back(1);
      rom_addr.push_back(1);
    }
  }
  for (size_t ilat = 0; ilat < lats.size(); ilat++) {
    if ((int)ilat == missing_layer)  // only applies to 3-hit, as in 4-hit missL=-1
      continue;
    auto lat = lats[ilat];
    if (lat == -1)
      lat = 0;
    rom_addr.push_back(lat);
  }
  std::reverse(rom_addr.begin(), rom_addr.end());
  return vhdl_unsigned_to_int(rom_addr);
}
