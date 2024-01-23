#include "L1Trigger/DTTriggerPhase2/interface/MuonPathCorFitter.h"
#include <cmath>
#include <memory>

using namespace edm;
using namespace std;
using namespace cmsdt;
// ============================================================================
// Constructors and destructor
// ============================================================================
MuonPathCorFitter::MuonPathCorFitter(const ParameterSet& pset,
                                     edm::ConsumesCollector& iC,
                                     std::shared_ptr<GlobalCoordsObtainer>& globalcoordsobtainer)
    : MuonPathFitter(pset, iC, globalcoordsobtainer), dT0_correlate_TP_(pset.getParameter<double>("dT0_correlate_TP")) {
  if (debug_)
    LogDebug("MuonPathCorFitter") << "MuonPathCorFitter: constructor";

  // LUTs
  both_sl_filename_ = pset.getParameter<edm::FileInPath>("lut_2sl");

  fillLuts();

  setChi2Th(pset.getParameter<double>("chi2corTh"));
  setTanPhiTh(pset.getParameter<double>("dTanPsi_correlate_TP"));
}

MuonPathCorFitter::~MuonPathCorFitter() {
  if (debug_)
    LogDebug("MuonPathCorFitter") << "MuonPathCorFitter: destructor";
}

// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void MuonPathCorFitter::initialise(const edm::EventSetup& iEventSetup) {
  if (debug_)
    LogDebug("MuonPathCorFitter") << "MuonPathCorFitter::initialiase";

  auto geom = iEventSetup.getHandle(dtGeomH);
  dtGeo_ = &(*geom);
}

void MuonPathCorFitter::run(edm::Event& iEvent,
                            const edm::EventSetup& iEventSetup,
                            std::vector<cmsdt::metaPrimitive>& inMPaths,
                            std::vector<cmsdt::metaPrimitive>& outMPaths) {
  if (debug_)
    LogDebug("MuonPathCorFitter") << "MuonPathCorFitter: run";
  if (!inMPaths.empty()) {
    int dum_sl_rawid = inMPaths[0].rawId;
    DTSuperLayerId dumSlId(dum_sl_rawid);
    DTChamberId ChId(dumSlId.wheel(), dumSlId.station(), dumSlId.sector());
    max_drift_tdc = maxdriftinfo_[dumSlId.wheel() + 2][dumSlId.station() - 1][dumSlId.sector() - 1];
    DTSuperLayerId sl1Id(ChId.rawId(), 1);
    DTSuperLayerId sl3Id(ChId.rawId(), 3);

    std::map<int, std::vector<metaPrimitive>> SL1metaPrimitivesPerBX;
    std::map<int, std::vector<metaPrimitive>> SL3metaPrimitivesPerBX;
    for (const auto& metaprimitiveIt : inMPaths) {
      int BX = metaprimitiveIt.t0 / 25;
      if (metaprimitiveIt.rawId == sl1Id.rawId())
        SL1metaPrimitivesPerBX[BX].push_back(metaprimitiveIt);
      else if (metaprimitiveIt.rawId == sl3Id.rawId())
        SL3metaPrimitivesPerBX[BX].push_back(metaprimitiveIt);
    }

    std::vector<bx_sl_vector> bxs_to_consider;
    bxs_to_consider.reserve(SL1metaPrimitivesPerBX.size());
    for (auto& prims_sl1 : SL1metaPrimitivesPerBX)
      bxs_to_consider.push_back(bx_sl_vector({prims_sl1.first, prims_sl1.second, 1}));

    for (auto& prims_sl3 : SL3metaPrimitivesPerBX)
      bxs_to_consider.push_back(bx_sl_vector({prims_sl3.first, prims_sl3.second, 3}));

    std::stable_sort(bxs_to_consider.begin(), bxs_to_consider.end(), bxSort);

    std::vector<mp_group> mps_q8;
    std::vector<mp_group> mps_q7;
    std::vector<mp_group> mps_q6;

    for (size_t ibx = 1; ibx < bxs_to_consider.size(); ibx++) {
      for (size_t ibx2 = 0; ibx2 < ibx; ibx2++) {
        if (bxs_to_consider[ibx].sl != bxs_to_consider[ibx2].sl &&
            (abs(bxs_to_consider[ibx].bx - bxs_to_consider[ibx2].bx)) <= MAX_BX_FOR_COR) {
          int isl1 = 0;
          for (auto& prim1 : bxs_to_consider[ibx].mps) {
            if (isl1 >= MAX_PRIM_PER_BX_FOR_COR)
              break;
            int isl2 = 0;
            for (auto& prim2 : bxs_to_consider[ibx2].mps) {
              if (isl2 >= MAX_PRIM_PER_BX_FOR_COR)
                break;
              if (bxs_to_consider[ibx].sl == 1) {
                if (!canCorrelate(prim1, prim2)) {
                  continue;
                }
                if (prim1.quality >= 3 && prim2.quality >= 3)
                  mps_q8.push_back(mp_group({prim1, prim2}));
                else if ((prim1.quality >= 3 && prim2.quality < 3) || (prim1.quality < 3 && prim2.quality >= 3))
                  mps_q7.push_back(mp_group({prim1, prim2}));
                else
                  mps_q6.push_back(mp_group({prim1, prim2}));
              } else {
                if (!canCorrelate(prim2, prim1)) {
                  continue;
                }
                if (prim2.quality >= 3 && prim1.quality >= 3)
                  mps_q8.push_back(mp_group({prim2, prim1}));
                else if ((prim2.quality >= 3 && prim1.quality < 3) || (prim2.quality < 3 && prim1.quality >= 3))
                  mps_q7.push_back(mp_group({prim2, prim1}));
                else
                  mps_q6.push_back(mp_group({prim2, prim1}));
              }
              isl2++;
            }
            isl1++;
          }
        }
      }  // looping over the 0 -> N-1 BX groups
    }    // looping over the 1 -> N BX groups
    int iq = 0;
    for (size_t i = 0; i < mps_q8.size(); i++) {
      if (iq >= MAX_PRIM_FOR_COR)
        break;
      analyze(mps_q8[i], outMPaths);
      iq += 1;
    }
    for (size_t i = 0; i < mps_q7.size(); i++) {
      if (iq >= MAX_PRIM_FOR_COR)
        break;
      analyze(mps_q7[i], outMPaths);
      iq += 1;
    }
    for (size_t i = 0; i < mps_q6.size(); i++) {
      if (iq >= MAX_PRIM_FOR_COR)
        break;
      analyze(mps_q6[i], outMPaths);
      iq += 1;
    }
  }
}

bool MuonPathCorFitter::canCorrelate(cmsdt::metaPrimitive mp_sl1, cmsdt::metaPrimitive mp_sl3) {
  // moving position from SL RF to chamber RF
  float pos_ch_sl1_f = mp_sl1.x;
  float pos_ch_sl3_f = mp_sl3.x;

  // translating into tdc counts
  int pos_ch_sl1 = int(pos_ch_sl1_f);
  int pos_ch_sl3 = int(pos_ch_sl3_f);

  int slope_sl1 = (int)mp_sl1.tanPhi;
  int slope_sl3 = (int)mp_sl3.tanPhi;

  if (abs((slope_sl1 >> WIDTH_POS_SLOPE_CORR) - (slope_sl3 >> WIDTH_POS_SLOPE_CORR)) > 1)
    return false;

  if (abs((pos_ch_sl1 >> WIDTH_POS_SLOPE_CORR) - (pos_ch_sl3 >> WIDTH_POS_SLOPE_CORR)) > 1)
    return false;

  if (abs(mp_sl1.t0 - mp_sl3.t0) > dT0_correlate_TP_)
    return false;

  return true;
}

void MuonPathCorFitter::finish() {
  if (debug_)
    LogDebug("MuonPathCorFitter") << "MuonPathCorFitter: finish";
};

//------------------------------------------------------------------
//--- Metodos privados
//------------------------------------------------------------------

void MuonPathCorFitter::analyze(mp_group mp, std::vector<cmsdt::metaPrimitive>& metaPrimitives) {
  //FIXME
  DTSuperLayerId MuonPathSLId(mp[0].rawId);  // SL1

  DTChamberId ChId(MuonPathSLId.wheel(), MuonPathSLId.station(), MuonPathSLId.sector());

  DTSuperLayerId MuonPathSL1Id(ChId.wheel(), ChId.station(), ChId.sector(), 1);
  DTSuperLayerId MuonPathSL3Id(ChId.wheel(), ChId.station(), ChId.sector(), 3);
  DTWireId wireIdSL1(MuonPathSL1Id, 2, 1);
  DTWireId wireIdSL3(MuonPathSL3Id, 2, 1);
  auto sl_shift_cm = shiftinfo_[wireIdSL1.rawId()] - shiftinfo_[wireIdSL3.rawId()];

  fit_common_in_t fit_common_in;

  // 8-element vectors, for the 8 layers. As here we are fitting one SL only, we leave the other SL values as dummy ones
  fit_common_in.hits = {};
  fit_common_in.hits_valid = {};
  short quality = 0;
  if (mp[0].quality >= 3 && mp[1].quality >= 3)
    quality = 8;
  else if ((mp[0].quality >= 3 && mp[1].quality < 3) || (mp[0].quality < 3 && mp[1].quality >= 3))
    quality = 7;
  else
    quality = 6;

  std::vector<int> missing_layers;

  for (int isl = 0; isl < 2; isl++) {
    int wire[4], tdc[4];
    if (isl != 1) {
      wire[0] = mp[isl].wi1;
      tdc[0] = mp[isl].tdc1;
      wire[1] = mp[isl].wi2;
      tdc[1] = mp[isl].tdc2;
      wire[2] = mp[isl].wi3;
      tdc[2] = mp[isl].tdc3;
      wire[3] = mp[isl].wi4;
      tdc[3] = mp[isl].tdc4;
    } else {
      wire[0] = mp[isl].wi5;
      tdc[0] = mp[isl].tdc5;
      wire[1] = mp[isl].wi6;
      tdc[1] = mp[isl].tdc6;
      wire[2] = mp[isl].wi7;
      tdc[2] = mp[isl].tdc7;
      wire[3] = mp[isl].wi8;
      tdc[3] = mp[isl].tdc8;
    }

    for (int i = 0; i < NUM_LAYERS; i++) {
      if (wire[i] != -1) {
        // Include both valid and non-valid hits. Non-valid values can be whatever, leaving all as -1 to make debugging easier.
        auto ti = tdc[i];
        if (ti != -1)
          ti = (int)round(((float)TIME_TO_TDC_COUNTS / (float)LHC_CLK_FREQ) * ti);
        auto wi = wire[i];
        auto ly = i;

        int wp_semicells = (wi - SL1_CELLS_OFFSET) * 2 + 1;
        if (ly % 2 == 1)
          wp_semicells -= 1;
        if (isl == 1)  // SL3
          wp_semicells -= (int)round((sl_shift_cm * 10) / CELL_SEMILENGTH);
        float wp_tdc = wp_semicells * max_drift_tdc;
        int wp = (int)((long int)(round(wp_tdc * std::pow(2, WIREPOS_WIDTH))) / (int)std::pow(2, WIREPOS_WIDTH));

        // wp in tdc counts (still in floating point)
        fit_common_in.hits.push_back({ti, wi, ly, wp});
        // fill valids as well
        fit_common_in.hits_valid.push_back(1);
      } else {
        missing_layers.push_back(isl * NUM_LAYERS + i);
        fit_common_in.hits.push_back({-1, -1, -1, -1});
        fit_common_in.hits_valid.push_back(0);
      }
    }
  }

  int smallest_time = 999999, tmp_coarse_wirepos_1 = -1, tmp_coarse_wirepos_3 = -1;
  // coarse_bctr is the 12 MSB of the smallest tdc
  for (int isl = 0; isl < 2; isl++) {
    for (size_t i = 0; i < NUM_LAYERS; i++) {
      if (fit_common_in.hits_valid[NUM_LAYERS * isl + i] == 0)
        continue;
      else if (fit_common_in.hits[NUM_LAYERS * isl + i].ti < smallest_time)
        smallest_time = fit_common_in.hits[NUM_LAYERS * isl + i].ti;
    }
  }
  if (fit_common_in.hits_valid[NUM_LAYERS * 0 + 0] == 1)
    tmp_coarse_wirepos_1 = fit_common_in.hits[NUM_LAYERS * 0 + 0].wp;
  else
    tmp_coarse_wirepos_1 = fit_common_in.hits[NUM_LAYERS * 0 + 1].wp;
  if (fit_common_in.hits_valid[NUM_LAYERS * 1 + 3] == 1)
    tmp_coarse_wirepos_3 = fit_common_in.hits[NUM_LAYERS * 1 + 3].wp;
  else
    tmp_coarse_wirepos_3 = fit_common_in.hits[NUM_LAYERS * 1 + 2].wp;

  tmp_coarse_wirepos_1 = tmp_coarse_wirepos_1 >> WIREPOS_NORM_LSB_IGNORED;
  tmp_coarse_wirepos_3 = tmp_coarse_wirepos_3 >> WIREPOS_NORM_LSB_IGNORED;

  fit_common_in.coarse_bctr = smallest_time >> (WIDTH_FULL_TIME - WIDTH_COARSED_TIME);
  fit_common_in.coarse_wirepos = (tmp_coarse_wirepos_1 + tmp_coarse_wirepos_3) >> 1;

  fit_common_in.lateralities.clear();

  auto rom_addr = get_rom_addr(mp, missing_layers);

  coeffs_t coeffs =
      RomDataConvert(lut_2sl[rom_addr], COEFF_WIDTH_COR_T0, COEFF_WIDTH_COR_POSITION, COEFF_WIDTH_COR_SLOPE, 0, 7);

  // Filling lateralities
  for (int isl = 0; isl < 2; isl++) {
    int lat[4];
    if (isl != 1) {
      lat[0] = mp[isl].lat1;
      lat[1] = mp[isl].lat2;
      lat[2] = mp[isl].lat3;
      lat[3] = mp[isl].lat4;
    } else {
      lat[0] = mp[isl].lat5;
      lat[1] = mp[isl].lat6;
      lat[2] = mp[isl].lat7;
      lat[3] = mp[isl].lat8;
    }

    for (size_t i = 0; i < NUM_LAYERS; i++) {
      fit_common_in.lateralities.push_back(lat[i]);
    }
  }

  fit_common_in.coeffs = coeffs;

  auto fit_common_out = fit(fit_common_in,
                            XI_COR_WIDTH,
                            COEFF_WIDTH_COR_T0,
                            COEFF_WIDTH_COR_POSITION,
                            COEFF_WIDTH_COR_SLOPE,
                            PRECISSION_COR_T0,
                            PRECISSION_COR_POSITION,
                            PRECISSION_COR_SLOPE,
                            PROD_RESIZE_COR_T0,
                            PROD_RESIZE_COR_POSITION,
                            PROD_RESIZE_COR_SLOPE,
                            max_drift_tdc,
                            0);

  if (fit_common_out.valid_fit == 1) {
    float t0_f = ((float)fit_common_out.t0) * (float)LHC_CLK_FREQ / (float)TIME_TO_TDC_COUNTS;
    float slope_f = -fit_common_out.slope * ((float)CELL_SEMILENGTH / max_drift_tdc) * (1) / (CELL_SEMIHEIGHT * 16.);
    if (std::abs(slope_f) > tanPhiTh_)
      return;

    DTWireId wireId(MuonPathSLId, 2, 1);
    float pos_ch_f = (float)(fit_common_out.position) * ((float)CELL_SEMILENGTH / (float)max_drift_tdc) / 10;
    pos_ch_f += (SL1_CELLS_OFFSET * CELL_LENGTH) / 10.;
    pos_ch_f += shiftinfo_[wireId.rawId()];

    float chi2_f = fit_common_out.chi2 * std::pow(((float)CELL_SEMILENGTH / (float)max_drift_tdc), 2) / 100;

    // obtention of global coordinates using luts
    int pos = (int)(10 * (pos_ch_f - shiftinfo_[wireId.rawId()]) * INCREASED_RES_POS_POW);
    int slope = (int)(-slope_f * INCREASED_RES_SLOPE_POW);
    auto global_coords = globalcoordsobtainer_->get_global_coordinates(ChId.rawId(), 0, pos, slope);
    float phi = global_coords[0];
    float phiB = global_coords[1];

    // obtention of global coordinates using cmssw geometry
    double z = 0;
    if (ChId.station() == 3 or ChId.station() == 4) {
      z += Z_SHIFT_MB4;
    }
    GlobalPoint jm_x_cmssw_global = dtGeo_->chamber(ChId)->toGlobal(LocalPoint(pos_ch_f, 0., z));
    int thisec = ChId.sector();
    if (thisec == 13)
      thisec = 4;
    if (thisec == 14)
      thisec = 10;
    float phi_cmssw = jm_x_cmssw_global.phi() - PHI_CONV * (thisec - 1);
    float psi = atan(slope_f);
    float phiB_cmssw = hasPosRF(ChId.wheel(), ChId.sector()) ? psi - phi_cmssw : -psi - phi_cmssw;
    metaPrimitives.emplace_back(metaPrimitive({MuonPathSLId.rawId(),
                                               t0_f,
                                               (double)fit_common_out.position,
                                               (double)fit_common_out.slope,
                                               phi,
                                               phiB,
                                               phi_cmssw,
                                               phiB_cmssw,
                                               chi2_f,
                                               quality,
                                               mp[0].wi1,
                                               mp[0].tdc1,
                                               mp[0].lat1,
                                               mp[0].wi2,
                                               mp[0].tdc2,
                                               mp[0].lat2,
                                               mp[0].wi3,
                                               mp[0].tdc3,
                                               mp[0].lat3,
                                               mp[0].wi4,
                                               mp[0].tdc4,
                                               mp[0].lat4,
                                               mp[1].wi5,
                                               mp[1].tdc5,
                                               mp[1].lat5,
                                               mp[1].wi6,
                                               mp[1].tdc6,
                                               mp[1].lat6,
                                               mp[1].wi7,
                                               mp[1].tdc7,
                                               mp[1].lat7,
                                               mp[1].wi8,
                                               mp[1].tdc8,
                                               mp[1].lat8,
                                               -1}));
  }
  return;
}

void MuonPathCorFitter::fillLuts() {
  std::ifstream ifin2sl(both_sl_filename_.fullPath());
  std::string line;
  while (ifin2sl.good()) {
    ifin2sl >> line;

    std::vector<int> myNumbers;
    for (size_t i = 0; i < line.size(); i++) {
      // This converts the char into an int and pushes it into vec
      myNumbers.push_back(line[i] - '0');  // The digits will be in the same order as before
    }
    std::reverse(myNumbers.begin(), myNumbers.end());
    lut_2sl.push_back(myNumbers);
  }

  return;
}

int MuonPathCorFitter::get_rom_addr(mp_group mps, std::vector<int> missing_hits) {
  std::vector<int> lats = {
      mps[0].lat1, mps[0].lat2, mps[0].lat3, mps[0].lat4, mps[1].lat5, mps[1].lat6, mps[1].lat7, mps[1].lat8};

  std::vector<int> rom_addr;
  if (missing_hits.size() == 1)
    rom_addr.push_back(1);
  else
    rom_addr.push_back(0);

  if (missing_hits.size() == 1) {  // 7 layers fit
    if (missing_hits[0] < 4)
      rom_addr.push_back(0);  // First SL has 4 hits (1) or 3 (0)
    else
      rom_addr.push_back(1);
    if (missing_hits[0] % 4 == 0) {
      rom_addr.push_back(0);
      rom_addr.push_back(0);
    } else if (missing_hits[0] % 4 == 1) {
      rom_addr.push_back(0);
      rom_addr.push_back(1);
    } else if (missing_hits[0] % 4 == 2) {
      rom_addr.push_back(1);
      rom_addr.push_back(0);
    } else {  // missing_hits[0] == 3
      rom_addr.push_back(1);
      rom_addr.push_back(1);
    }
    for (size_t ilat = 0; ilat < lats.size(); ilat++) {
      if ((int)ilat == missing_hits[0])  // only applies to 3-hit, as in 4-hit missL=-1
        continue;
      auto lat = lats[ilat];
      if (lat == -1)
        lat = 0;
      rom_addr.push_back(lat);
    }

  } else if (missing_hits.empty()) {  // 8 layers fit
    for (size_t ilat = 0; ilat < lats.size(); ilat++) {
      auto lat = lats[ilat];
      if (lat == -1)
        lat = 0;
      rom_addr.push_back(lat);
    }
    auto lat = lats[NUM_LAYERS + 3];
    if (lat == -1)
      lat = 0;
    rom_addr.push_back(lat);
    rom_addr.push_back(lat);

  } else {  // 6 layers fit
    for (int i = missing_hits.size() - 1; i >= 0; i--) {
      if (missing_hits[i] % 4 == 0) {
        rom_addr.push_back(0);
        rom_addr.push_back(0);
      } else if (missing_hits[i] % 4 == 1) {
        rom_addr.push_back(0);
        rom_addr.push_back(1);
      } else if (missing_hits[i] % 4 == 2) {
        rom_addr.push_back(1);
        rom_addr.push_back(0);
      } else {  // missing_hits[i] % 4 == 3
        rom_addr.push_back(1);
        rom_addr.push_back(1);
      }
    }
    for (size_t ilat = 0; ilat < lats.size(); ilat++) {
      if ((int)ilat == missing_hits[0] || (int)ilat == missing_hits[1])  // only applies to 3-hit, as in 4-hit missL=-1
        continue;
      auto lat = lats[ilat];
      if (lat == -1)
        lat = 0;
      rom_addr.push_back(lat);
    }
  }
  std::reverse(rom_addr.begin(), rom_addr.end());
  return vhdl_unsigned_to_int(rom_addr);
}
