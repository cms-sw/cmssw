#include "L1Trigger/DTTriggerPhase2/interface/MuonPathAnalyticAnalyzer.h"
#include <cmath>
#include <memory>

using namespace edm;
using namespace std;
using namespace cmsdt;
// ============================================================================
// Constructors and destructor
// ============================================================================
MuonPathAnalyticAnalyzer::MuonPathAnalyticAnalyzer(const ParameterSet &pset,
                                                   edm::ConsumesCollector &iC,
                                                   std::shared_ptr<GlobalCoordsObtainer> &globalcoordsobtainer)
    : MuonPathAnalyzer(pset, iC),
      debug_(pset.getUntrackedParameter<bool>("debug")),
      chi2Th_(pset.getParameter<double>("chi2Th")),
      tanPhiTh_(pset.getParameter<double>("tanPhiTh")),
      tanPhiThw2max_(pset.getParameter<double>("tanPhiThw2max")),
      tanPhiThw2min_(pset.getParameter<double>("tanPhiThw2min")),
      tanPhiThw1max_(pset.getParameter<double>("tanPhiThw1max")),
      tanPhiThw1min_(pset.getParameter<double>("tanPhiThw1min")),
      tanPhiThw0_(pset.getParameter<double>("tanPhiThw0")) {
  if (debug_)
    LogDebug("MuonPathAnalyticAnalyzer") << "MuonPathAnalyzer: constructor";

  fillLAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER();

  //shift phi
  int rawId;
  shift_filename_ = pset.getParameter<edm::FileInPath>("shift_filename");
  std::ifstream ifin3(shift_filename_.fullPath());
  double shift;
  if (ifin3.fail()) {
    throw cms::Exception("Missing Input File")
        << "MuonPathAnalyticAnalyzer::MuonPathAnalyticAnalyzer() -  Cannot find " << shift_filename_.fullPath();
  }
  while (ifin3.good()) {
    ifin3 >> rawId >> shift;
    shiftinfo_[rawId] = shift;
  }

  //shift theta

  shift_theta_filename_ = pset.getParameter<edm::FileInPath>("shift_theta_filename");
  std::ifstream ifin4(shift_theta_filename_.fullPath());
  if (ifin4.fail()) {
    throw cms::Exception("Missing Input File")
        << "MuonPathAnalyzerPerSL::MuonPathAnalyzerPerSL() -  Cannot find " << shift_theta_filename_.fullPath();
  }

  while (ifin4.good()) {
    ifin4 >> rawId >> shift;
    shiftthetainfo_[rawId] = shift;
  }

  chosen_sl_ = pset.getParameter<int>("trigger_with_sl");

  if (chosen_sl_ != 1 && chosen_sl_ != 3 && chosen_sl_ != 4) {
    LogDebug("MuonPathAnalyticAnalyzer") << "chosen sl must be 1,3 or 4(both superlayers)";
    assert(chosen_sl_ != 1 && chosen_sl_ != 3 && chosen_sl_ != 4);  //4 means run using the two superlayers
  }

  dtGeomH = iC.esConsumes<DTGeometry, MuonGeometryRecord, edm::Transition::BeginRun>();
  globalcoordsobtainer_ = globalcoordsobtainer;
}

MuonPathAnalyticAnalyzer::~MuonPathAnalyticAnalyzer() {
  if (debug_)
    LogDebug("MuonPathAnalyticAnalyzer") << "MuonPathAnalyzer: destructor";
}

// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void MuonPathAnalyticAnalyzer::initialise(const edm::EventSetup &iEventSetup) {
  if (debug_)
    LogDebug("MuonPathAnalyticAnalyzer") << "MuonPathAnalyticAnalyzer::initialiase";

  auto geom = iEventSetup.getHandle(dtGeomH);
  dtGeo_ = &(*geom);
}

void MuonPathAnalyticAnalyzer::run(edm::Event &iEvent,
                                   const edm::EventSetup &iEventSetup,
                                   MuonPathPtrs &muonpaths,
                                   std::vector<metaPrimitive> &metaPrimitives) {
  if (debug_)
    LogDebug("MuonPathAnalyticAnalyzer") << "MuonPathAnalyticAnalyzer: run";

  // fit per SL (need to allow for multiple outputs for a single mpath)
  for (auto &muonpath : muonpaths) {
    analyze(muonpath, metaPrimitives);
  }
}

void MuonPathAnalyticAnalyzer::finish() {
  if (debug_)
    LogDebug("MuonPathAnalyticAnalyzer") << "MuonPathAnalyzer: finish";
};

//------------------------------------------------------------------
//--- Metodos privados
//------------------------------------------------------------------

void MuonPathAnalyticAnalyzer::analyze(MuonPathPtr &inMPath, std::vector<metaPrimitive> &metaPrimitives) {
  if (debug_)
    LogDebug("MuonPathAnalyticAnalyzer") << "DTp2:analyze \t\t\t\t starts";
  // LOCATE MPATH
  int selected_Id = 0;
  if (inMPath->primitive(0)->tdcTimeStamp() != -1)
    selected_Id = inMPath->primitive(0)->cameraId();
  else if (inMPath->primitive(1)->tdcTimeStamp() != -1)
    selected_Id = inMPath->primitive(1)->cameraId();
  else if (inMPath->primitive(2)->tdcTimeStamp() != -1)
    selected_Id = inMPath->primitive(2)->cameraId();
  else if (inMPath->primitive(3)->tdcTimeStamp() != -1)
    selected_Id = inMPath->primitive(3)->cameraId();

  DTLayerId thisLId(selected_Id);
  if (debug_)
    LogDebug("MuonPathAnalyticAnalyzer") << "Building up MuonPathSLId from rawId in the Primitive";
  DTSuperLayerId MuonPathSLId(thisLId.wheel(), thisLId.station(), thisLId.sector(), thisLId.superLayer());
  if (debug_)
    LogDebug("MuonPathAnalyticAnalyzer") << "The MuonPathSLId is" << MuonPathSLId;

  if (debug_)
    LogDebug("MuonPathAnalyticAnalyzer")
        << "DTp2:analyze \t\t\t\t In analyze function checking if inMPath->isAnalyzable() " << inMPath->isAnalyzable();

  if (chosen_sl_ < 4 && thisLId.superLayer() != chosen_sl_)
    return;  // avoid running when mpath not in chosen SL (for 1SL fitting)

  auto mPath = std::make_shared<MuonPath>(inMPath);
  mPath->setQuality(NOPATH);

  int wi[4], wires[4], t0s[4], valids[4];
  // bool is_four_hit = true;
  for (int j = 0; j < NUM_LAYERS; j++) {
    if (mPath->primitive(j)->isValidTime()) {
      wi[j] = mPath->primitive(j)->channelId();
      wires[j] = mPath->primitive(j)->channelId();
      t0s[j] = mPath->primitive(j)->tdcTimeStamp();
      valids[j] = 1;
    } else {
      wi[j] = -1;
      wires[j] = -1;
      t0s[j] = -1;
      valids[j] = 0;
      // is_four_hit = false;
    }
  }

  if (wi[0] < 0)
    wi[0] = wi[1];
  else if (wi[1] < 0)
    wi[1] = wi[0];
  else if (wi[2] < 0)
    wi[2] = wi[1] - 1;
  else if (wi[3] < 0)
    wi[3] = wi[2];

  int cell_horiz_layout[4];
  for (int lay = 0; lay < NUM_LAYERS; lay++) {
    cell_horiz_layout[lay] = (wi[lay] - wi[0]) * 2;
    if (lay % 2 != 0)
      cell_horiz_layout[lay]--;
  }

  // calculate the coarse offset position
  int tmp = 1;
  if (valids[1] == 0)
    tmp = 3;
  int coarse_pos = (wi[tmp] * 2 - cell_horiz_layout[tmp]) * 21 * std::pow(2, 4);

  //calculate the relative position of wires in mm wrt layer 0's cell wire
  int xwire_mm[4];
  for (int lay = 0; lay < NUM_LAYERS; lay++) {
    xwire_mm[lay] = 21 * cell_horiz_layout[lay];
  }

  // divide the timestamps in coarse + reduced part
  int valid_coarse_times[4], min_coarse_time = 999999, max_coarse_time = -999999;
  for (int lay = 0; lay < NUM_LAYERS; lay++) {
    if (valids[lay] == 1) {
      valid_coarse_times[lay] = (t0s[lay] >> (TDCTIME_REDUCED_SIZE - 1));
      if (valid_coarse_times[lay] < min_coarse_time) {
        min_coarse_time = valid_coarse_times[lay];
      }
      if (valid_coarse_times[lay] > max_coarse_time) {
        max_coarse_time = valid_coarse_times[lay];
      }
    } else {
      valid_coarse_times[lay] = -1;
    }
  }

  if (max_coarse_time - min_coarse_time >= 2)
    return;
  int coarse_offset = max_coarse_time - 1;

  int reduced_times[4];
  for (int lay = 0; lay < NUM_LAYERS; lay++) {
    reduced_times[lay] =
        ((1 - ((max_coarse_time & 1) ^ ((t0s[lay] >> (TDCTIME_REDUCED_SIZE - 1)) & 1))) << (TDCTIME_REDUCED_SIZE - 1));
    reduced_times[lay] += (t0s[lay] & std::stoi(std::string(TDCTIME_REDUCED_SIZE - 1, '1'), nullptr, 2));
  }
  std::vector<LATCOMB_CONSTANTS> latcomb_consts_arr;
  for (auto &elem : LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER)
    if (elem.cell_valid_layout.valid[0] == valids[0] && elem.cell_valid_layout.valid[1] == valids[1] &&
        elem.cell_valid_layout.valid[2] == valids[2] && elem.cell_valid_layout.valid[3] == valids[3] &&
        elem.cell_valid_layout.cell_horiz_layout[0] == cell_horiz_layout[0] &&
        elem.cell_valid_layout.cell_horiz_layout[1] == cell_horiz_layout[1] &&
        elem.cell_valid_layout.cell_horiz_layout[2] == cell_horiz_layout[2] &&
        elem.cell_valid_layout.cell_horiz_layout[3] == cell_horiz_layout[3])
      for (auto &ind_latcomb_consts : elem.latcomb_constants)
        latcomb_consts_arr.push_back(ind_latcomb_consts);
  for (auto &latcomb_consts : latcomb_consts_arr) {
    segment_fitter(MuonPathSLId,
                   wires,
                   t0s,
                   valids,
                   reduced_times,
                   cell_horiz_layout,
                   latcomb_consts,
                   xwire_mm,
                   coarse_pos,
                   coarse_offset,
                   metaPrimitives);
  }
}

int MuonPathAnalyticAnalyzer::compute_parameter(MAGNITUDE constants, int t0s[4], int DIV_SHR_BITS, int INCREASED_RES) {
  long int result = 0;
  for (int lay = 0; lay < NUM_LAYERS; lay++) {
    result += constants.coeff[lay] * t0s[lay];
  }
  result = ((result * int(std::pow(2, INCREASED_RES)) + constants.add) * constants.mult) >> DIV_SHR_BITS;

  return result;
}

void MuonPathAnalyticAnalyzer::segment_fitter(DTSuperLayerId MuonPathSLId,
                                              int wires[4],
                                              int t0s[4],
                                              int valid[4],
                                              int reduced_times[4],
                                              int cell_horiz_layout[4],
                                              LATCOMB_CONSTANTS latcomb_consts,
                                              int xwire_mm[4],
                                              int coarse_pos,
                                              int coarse_offset,
                                              std::vector<cmsdt::metaPrimitive> &metaPrimitives) {
  auto latcomb = latcomb_consts.latcomb;
  auto constants = latcomb_consts.constants;
  bool is_four_hit = true;

  if (latcomb == 0)
    return;

  int lat_array[4];
  for (int lay = 0; lay < NUM_LAYERS; lay++) {
    if (((latcomb >> lay) & 1) != 0) {
      lat_array[lay] = 1;
    } else
      lat_array[lay] = -1;
  }

  int time = compute_parameter(constants.t0, reduced_times, DIV_SHR_BITS_T0, INCREASED_RES_T0);
  int pos = compute_parameter(constants.pos, reduced_times, DIV_SHR_BITS_POS, INCREASED_RES_POS);
  int slope = compute_parameter(constants.slope, reduced_times, DIV_SHR_BITS_SLOPE, INCREASED_RES_SLOPE);
  int slope_xhh =
      compute_parameter(constants.slope_xhh, reduced_times, DIV_SHR_BITS_SLOPE_XHH, INCREASED_RES_SLOPE_XHH);

  int bx_time = time + (coarse_offset << (TDCTIME_REDUCED_SIZE - 1));

  pos += coarse_pos;

  int chi2_mm2_p = 0;
  for (int lay = 0; lay < NUM_LAYERS; lay++) {
    int drift_time = reduced_times[lay] - time;
    if (valid[lay] == 1 && (drift_time < 0 || drift_time > MAXDRIFT))
      return;

    int drift_dist = ((((drift_time * INCREASED_RES_POS_POW) + DTDD_PREADD) * DTDD_MULT) >> DTDD_SHIFTR_BITS);
    int xdist = xwire_mm[lay] * pow(2, 4) - (pos - coarse_pos) + lat_array[lay] * drift_dist;
    xdist -= (3 - 2 * (3 - lay)) * slope_xhh;
    int res = xdist;
    if (valid[lay] == 0) {
      res = 0;
      is_four_hit = false;
    }
    chi2_mm2_p += res * res * 4;
  }

  int quality = HIGHQ;
  if (!is_four_hit)
    quality = LOWQ;

  // Obtain coordinate values in floating point
  double pos_f, slope_f, chi2_f;
  DTWireId wireId(MuonPathSLId, 2, 1);

  pos_f = double(pos) +
          int(10 * shiftinfo_[wireId.rawId()] * INCREASED_RES_POS_POW);  // position in mm * precision in JM RF
  pos_f /= (10. * INCREASED_RES_POS_POW);                                // position in cm w.r.t center of the chamber
  slope_f = -(double(slope) / INCREASED_RES_SLOPE_POW);
  chi2_f = double(chi2_mm2_p) / (16. * 64. * 100.);

  // Impose the thresholds
  if (MuonPathSLId.superLayer() != 2)
    if (std::abs(slope_f) > tanPhiTh_)
      return;
  if (chi2_f > (chi2Th_))
    return;

  // Compute phi and phib
  // Implemented using cmssw geometry and fw-like approach
  DTChamberId ChId(MuonPathSLId.wheel(), MuonPathSLId.station(), MuonPathSLId.sector());
  // fw-like variables
  double phi = -999.;
  double phiB = -999.;
  // cmssw-like variables
  double phi_cmssw = -999.;
  double phiB_cmssw = -999.;
  if (MuonPathSLId.superLayer() != 2) {
    double z = 0;
    double z1 = Z_POS_SL;
    double z3 = -1. * z1;
    if (ChId.station() == 3 or ChId.station() == 4) {
      z1 = z1 + Z_SHIFT_MB4;
      z3 = z3 + Z_SHIFT_MB4;
    }
    if (MuonPathSLId.superLayer() == 1)
      z = z1;
    else if (MuonPathSLId.superLayer() == 3)
      z = z3;

    // cmssw-like calculation
    GlobalPoint jm_x_cmssw_global = dtGeo_->chamber(ChId)->toGlobal(LocalPoint(pos_f, 0., z));
    int thisec = MuonPathSLId.sector();
    if (thisec == 13)
      thisec = 4;
    if (thisec == 14)
      thisec = 10;
    phi_cmssw = jm_x_cmssw_global.phi() - PHI_CONV * (thisec - 1);
    double psi = atan(slope_f);
    phiB_cmssw = hasPosRF(MuonPathSLId.wheel(), MuonPathSLId.sector()) ? psi - phi_cmssw : -psi - phi_cmssw;

    auto global_coords =
        globalcoordsobtainer_->get_global_coordinates(ChId.rawId(), MuonPathSLId.superLayer(), pos, slope);
    phi = global_coords[0];
    phiB = global_coords[1];
  } else {
    // Impose the thresholds
    if (std::abs(MuonPathSLId.wheel()) == 2) {
      if (slope_f > tanPhiThw2max_ or slope_f < tanPhiThw2min_)
        return;
    }
    if (std::abs(MuonPathSLId.wheel()) == 1) {
      if (slope_f > tanPhiThw1max_ or slope_f < tanPhiThw1min_)
        return;
    }
    if (MuonPathSLId.wheel() == 0) {
      if (std::abs(slope_f) > tanPhiThw0_)
        return;
    }

    // fw-like calculation
    DTLayerId SL2_layer2Id(MuonPathSLId, 2);
    double z_shift = shiftthetainfo_[SL2_layer2Id.rawId()];
    double jm_y = hasPosRF(MuonPathSLId.wheel(), MuonPathSLId.sector()) ? z_shift - pos_f : z_shift + pos_f;
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
  }

  // get the lateralities (in reverse order) in order to fill the metaprimitive
  std::vector<int> lateralities = getLateralityCombination(latcomb);
  for (int lay = 0; lay < NUM_LAYERS; lay++) {
    if (valid[lay] == 0)
      lateralities[lay] = -1;
  }

  metaPrimitives.emplace_back(metaPrimitive({MuonPathSLId.rawId(),
                                             double(bx_time),
                                             pos_f,
                                             slope_f,
                                             phi,
                                             phiB,
                                             phi_cmssw,
                                             phiB_cmssw,
                                             chi2_f,
                                             quality,
                                             wires[0],
                                             t0s[0],
                                             lateralities[0],
                                             wires[1],
                                             t0s[1],
                                             lateralities[1],
                                             wires[2],
                                             t0s[2],
                                             lateralities[2],
                                             wires[3],
                                             t0s[3],
                                             lateralities[3],
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

std::vector<int> MuonPathAnalyticAnalyzer::getLateralityCombination(int latcomb) {
  // returns the latcomb as a binary number represented in a vector of integers
  // careful, the output is in reverse order
  std::vector<int> binaryNum = {};
  while (latcomb > 1) {
    binaryNum.push_back(latcomb % 2);
    latcomb = latcomb / 2;
  }
  binaryNum.push_back(latcomb);
  while (binaryNum.size() < 4)
    binaryNum.push_back(0);
  return binaryNum;
}

void MuonPathAnalyticAnalyzer::fillLAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER() {
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, -1, 0, -1}, {1, 1, 0, 1}},
                                                    {
                                                        {1,
                                                         {
                                                             {-6170, {1, 0, 0, -1}, 56936},
                                                             {239, {0, 1, 0, -1}, 4380},
                                                             {37, {0, 1, 0, -1}, 3559},
                                                             {776, {2, 3, 0, -1}, 16384},
                                                         }},
                                                        {2,
                                                         {
                                                             {-30885, {-1, 3, 0, -2}, 18979},
                                                             {-1583769, {1, 0, 0, -1}, 2920},
                                                             {-6133, {1, 0, 0, -1}, 2372},
                                                             {-771, {2, 3, 0, 1}, 10923},
                                                         }},
                                                        {3,
                                                         {
                                                             {-6170, {1, 0, 0, -1}, 56936},
                                                             {-1584008, {-1, 1, 0, 0}, 8759},
                                                             {-6170, {-1, 1, 0, 0}, 7117},
                                                             {-773, {-2, 3, 0, 1}, 32768},
                                                         }},
                                                        {8,
                                                         {
                                                             {-6170, {-1, 0, 0, 1}, 56936},
                                                             {-1584008, {1, -1, 0, 0}, 8759},
                                                             {-6170, {1, -1, 0, 0}, 7117},
                                                             {775, {-2, 3, 0, 1}, 32768},
                                                         }},
                                                        {9,
                                                         {
                                                             {-30885, {1, -3, 0, 2}, 18979},
                                                             {-1583769, {-1, 0, 0, 1}, 2920},
                                                             {-6133, {-1, 0, 0, 1}, 2372},
                                                             {777, {2, 3, 0, 1}, 10923},
                                                         }},
                                                        {10,
                                                         {
                                                             {-6170, {-1, 0, 0, 1}, 56936},
                                                             {239, {0, -1, 0, 1}, 4380},
                                                             {37, {0, -1, 0, 1}, 3559},
                                                             {-772, {2, 3, 0, -1}, 16384},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, -1, 0, 1}, {0, 1, 1, 1}},
                                                    {
                                                        {2,
                                                         {
                                                             {-6170, {0, 1, -1, 0}, 56936},
                                                             {1584248, {0, 0, 1, -1}, 8759},
                                                             {6206, {0, 0, 1, -1}, 7117},
                                                             {1, {0, 1, 2, -1}, 32768},
                                                         }},
                                                        {4,
                                                         {
                                                             {-6170, {0, -1, 1, 0}, 56936},
                                                             {3168495, {0, 1, 0, -1}, 4380},
                                                             {12413, {0, 1, 0, -1}, 3559},
                                                             {2, {0, 1, 2, 1}, 16384},
                                                         }},
                                                        {6,
                                                         {
                                                             {-6170, {0, 2, -1, -1}, 56936},
                                                             {1584248, {0, -1, 1, 0}, 8759},
                                                             {6206, {0, -1, 1, 0}, 7117},
                                                             {1, {0, -1, 2, 1}, 32768},
                                                         }},
                                                        {8,
                                                         {
                                                             {-6170, {0, -2, 1, 1}, 56936},
                                                             {1584248, {0, 1, -1, 0}, 8759},
                                                             {6206, {0, 1, -1, 0}, 7117},
                                                             {1, {0, -1, 2, 1}, 32768},
                                                         }},
                                                        {10,
                                                         {
                                                             {-6170, {0, 1, -1, 0}, 56936},
                                                             {3168495, {0, -1, 0, 1}, 4380},
                                                             {12413, {0, -1, 0, 1}, 3559},
                                                             {2, {0, 1, 2, 1}, 16384},
                                                         }},
                                                        {12,
                                                         {
                                                             {-6170, {0, -1, 1, 0}, 56936},
                                                             {1584248, {0, 0, -1, 1}, 8759},
                                                             {6206, {0, 0, -1, 1}, 7117},
                                                             {1, {0, 1, 2, -1}, 32768},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, -1, -2, -3}, {1, 1, 0, 1}},
                                                    {
                                                        {1,
                                                         {
                                                             {-18546, {1, 0, 0, -1}, 56936},
                                                             {-3168017, {0, 1, 0, -1}, 4380},
                                                             {-12339, {0, 1, 0, -1}, 3559},
                                                             {2, {2, 3, 0, -1}, 16384},
                                                         }},
                                                        {2,
                                                         {
                                                             {-55637, {-1, 3, 0, -2}, 18979},
                                                             {-4752025, {1, 0, 0, -1}, 2920},
                                                             {-18509, {1, 0, 0, -1}, 2372},
                                                             {3, {2, 3, 0, 1}, 10923},
                                                         }},
                                                        {3,
                                                         {
                                                             {-18546, {1, 0, 0, -1}, 56936},
                                                             {-1584008, {-1, 1, 0, 0}, 8759},
                                                             {-6170, {-1, 1, 0, 0}, 7117},
                                                             {1, {-2, 3, 0, 1}, 32768},
                                                         }},
                                                        {8,
                                                         {
                                                             {-18546, {-1, 0, 0, 1}, 56936},
                                                             {-1584008, {1, -1, 0, 0}, 8759},
                                                             {-6170, {1, -1, 0, 0}, 7117},
                                                             {1, {-2, 3, 0, 1}, 32768},
                                                         }},
                                                        {9,
                                                         {
                                                             {-55637, {1, -3, 0, 2}, 18979},
                                                             {-4752025, {-1, 0, 0, 1}, 2920},
                                                             {-18509, {-1, 0, 0, 1}, 2372},
                                                             {3, {2, 3, 0, 1}, 10923},
                                                         }},
                                                        {10,
                                                         {
                                                             {-18546, {-1, 0, 0, 1}, 56936},
                                                             {-3168017, {0, -1, 0, 1}, 4380},
                                                             {-12339, {0, -1, 0, 1}, 3559},
                                                             {2, {2, 3, 0, -1}, 16384},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, 1, 0, 1}, {0, 1, 1, 1}},
                                                    {
                                                        {2,
                                                         {
                                                             {6206, {0, 1, -1, 0}, 56936},
                                                             {1584248, {0, 0, 1, -1}, 8759},
                                                             {6206, {0, 0, 1, -1}, 7117},
                                                             {775, {0, 1, 2, -1}, 32768},
                                                         }},
                                                        {4,
                                                         {
                                                             {6206, {0, -1, 1, 0}, 56936},
                                                             {239, {0, 1, 0, -1}, 4380},
                                                             {37, {0, 1, 0, -1}, 3559},
                                                             {-772, {0, 1, 2, 1}, 16384},
                                                         }},
                                                        {6,
                                                         {
                                                             {18582, {0, 2, -1, -1}, 56936},
                                                             {-1584008, {0, -1, 1, 0}, 8759},
                                                             {-6170, {0, -1, 1, 0}, 7117},
                                                             {-773, {0, -1, 2, 1}, 32768},
                                                         }},
                                                        {8,
                                                         {
                                                             {18582, {0, -2, 1, 1}, 56936},
                                                             {-1584008, {0, 1, -1, 0}, 8759},
                                                             {-6170, {0, 1, -1, 0}, 7117},
                                                             {775, {0, -1, 2, 1}, 32768},
                                                         }},
                                                        {10,
                                                         {
                                                             {6206, {0, 1, -1, 0}, 56936},
                                                             {239, {0, -1, 0, 1}, 4380},
                                                             {37, {0, -1, 0, 1}, 3559},
                                                             {776, {0, 1, 2, 1}, 16384},
                                                         }},
                                                        {12,
                                                         {
                                                             {6206, {0, -1, 1, 0}, 56936},
                                                             {1584248, {0, 0, -1, 1}, 8759},
                                                             {6206, {0, 0, -1, 1}, 7117},
                                                             {-773, {0, 1, 2, -1}, 32768},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, 1, 2, 1}, {1, 1, 1, 0}},
                                                    {
                                                        {1,
                                                         {
                                                             {18582, {1, 1, -2, 0}, 56936},
                                                             {1584248, {0, 1, -1, 0}, 8759},
                                                             {6206, {0, 1, -1, 0}, 7117},
                                                             {1, {1, 2, -1, 0}, 32768},
                                                         }},
                                                        {2,
                                                         {
                                                             {18582, {0, 1, -1, 0}, 56936},
                                                             {3168495, {1, 0, -1, 0}, 4380},
                                                             {12413, {1, 0, -1, 0}, 3559},
                                                             {2, {1, 2, 1, 0}, 16384},
                                                         }},
                                                        {3,
                                                         {
                                                             {18582, {0, 1, -1, 0}, 56936},
                                                             {1584248, {-1, 1, 0, 0}, 8759},
                                                             {6206, {-1, 1, 0, 0}, 7117},
                                                             {1, {-1, 2, 1, 0}, 32768},
                                                         }},
                                                        {4,
                                                         {
                                                             {18582, {0, -1, 1, 0}, 56936},
                                                             {1584248, {1, -1, 0, 0}, 8759},
                                                             {6206, {1, -1, 0, 0}, 7117},
                                                             {1, {-1, 2, 1, 0}, 32768},
                                                         }},
                                                        {5,
                                                         {
                                                             {18582, {0, -1, 1, 0}, 56936},
                                                             {3168495, {-1, 0, 1, 0}, 4380},
                                                             {12413, {-1, 0, 1, 0}, 3559},
                                                             {2, {1, 2, 1, 0}, 16384},
                                                         }},
                                                        {6,
                                                         {
                                                             {18582, {-1, -1, 2, 0}, 56936},
                                                             {1584248, {0, -1, 1, 0}, 8759},
                                                             {6206, {0, -1, 1, 0}, 7117},
                                                             {1, {1, 2, -1, 0}, 32768},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, 1, 0, -1}, {1, 0, 1, 1}},
                                                    {
                                                        {1,
                                                         {
                                                             {-6170, {1, 0, 0, -1}, 56936},
                                                             {-1584008, {0, 0, 1, -1}, 8759},
                                                             {-6170, {0, 0, 1, -1}, 7117},
                                                             {-773, {1, 0, 3, -2}, 32768},
                                                         }},
                                                        {4,
                                                         {
                                                             {-6133, {-2, 0, 3, -1}, 18979},
                                                             {-1583769, {1, 0, 0, -1}, 2920},
                                                             {-6133, {1, 0, 0, -1}, 2372},
                                                             {777, {1, 0, 3, 2}, 10923},
                                                         }},
                                                        {5,
                                                         {
                                                             {-6170, {1, 0, 0, -1}, 56936},
                                                             {239, {-1, 0, 1, 0}, 4380},
                                                             {37, {-1, 0, 1, 0}, 3559},
                                                             {776, {-1, 0, 3, 2}, 16384},
                                                         }},
                                                        {8,
                                                         {
                                                             {-6170, {-1, 0, 0, 1}, 56936},
                                                             {239, {1, 0, -1, 0}, 4380},
                                                             {37, {1, 0, -1, 0}, 3559},
                                                             {-772, {-1, 0, 3, 2}, 16384},
                                                         }},
                                                        {9,
                                                         {
                                                             {-6133, {2, 0, -3, 1}, 18979},
                                                             {-1583769, {-1, 0, 0, 1}, 2920},
                                                             {-6133, {-1, 0, 0, 1}, 2372},
                                                             {-771, {1, 0, 3, 2}, 10923},
                                                         }},
                                                        {12,
                                                         {
                                                             {-6170, {-1, 0, 0, 1}, 56936},
                                                             {-1584008, {0, 0, -1, 1}, 8759},
                                                             {-6170, {0, 0, -1, 1}, 7117},
                                                             {775, {1, 0, 3, -2}, 32768},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, -1, -2, -1}, {1, 1, 1, 0}},
                                                    {
                                                        {1,
                                                         {
                                                             {-18546, {1, 1, -2, 0}, 56936},
                                                             {-1584008, {0, 1, -1, 0}, 8759},
                                                             {-6170, {0, 1, -1, 0}, 7117},
                                                             {1, {1, 2, -1, 0}, 32768},
                                                         }},
                                                        {2,
                                                         {
                                                             {-18546, {0, 1, -1, 0}, 56936},
                                                             {-3168017, {1, 0, -1, 0}, 4380},
                                                             {-12339, {1, 0, -1, 0}, 3559},
                                                             {2, {1, 2, 1, 0}, 16384},
                                                         }},
                                                        {3,
                                                         {
                                                             {-18546, {0, 1, -1, 0}, 56936},
                                                             {-1584008, {-1, 1, 0, 0}, 8759},
                                                             {-6170, {-1, 1, 0, 0}, 7117},
                                                             {1, {-1, 2, 1, 0}, 32768},
                                                         }},
                                                        {4,
                                                         {
                                                             {-18546, {0, -1, 1, 0}, 56936},
                                                             {-1584008, {1, -1, 0, 0}, 8759},
                                                             {-6170, {1, -1, 0, 0}, 7117},
                                                             {1, {-1, 2, 1, 0}, 32768},
                                                         }},
                                                        {5,
                                                         {
                                                             {-18546, {0, -1, 1, 0}, 56936},
                                                             {-3168017, {-1, 0, 1, 0}, 4380},
                                                             {-12339, {-1, 0, 1, 0}, 3559},
                                                             {2, {1, 2, 1, 0}, 16384},
                                                         }},
                                                        {6,
                                                         {
                                                             {-18546, {-1, -1, 2, 0}, 56936},
                                                             {-1584008, {0, -1, 1, 0}, 8759},
                                                             {-6170, {0, -1, 1, 0}, 7117},
                                                             {1, {1, 2, -1, 0}, 32768},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, -1, -2, -3}, {0, 1, 1, 1}},
                                                    {
                                                        {2,
                                                         {
                                                             {-18546, {0, 1, -1, 0}, 56936},
                                                             {-1584008, {0, 0, 1, -1}, 8759},
                                                             {-6170, {0, 0, 1, -1}, 7117},
                                                             {1, {0, 1, 2, -1}, 32768},
                                                         }},
                                                        {4,
                                                         {
                                                             {-18546, {0, -1, 1, 0}, 56936},
                                                             {-3168017, {0, 1, 0, -1}, 4380},
                                                             {-12339, {0, 1, 0, -1}, 3559},
                                                             {2, {0, 1, 2, 1}, 16384},
                                                         }},
                                                        {6,
                                                         {
                                                             {-18546, {0, 2, -1, -1}, 56936},
                                                             {-1584008, {0, -1, 1, 0}, 8759},
                                                             {-6170, {0, -1, 1, 0}, 7117},
                                                             {1, {0, -1, 2, 1}, 32768},
                                                         }},
                                                        {8,
                                                         {
                                                             {-18546, {0, -2, 1, 1}, 56936},
                                                             {-1584008, {0, 1, -1, 0}, 8759},
                                                             {-6170, {0, 1, -1, 0}, 7117},
                                                             {1, {0, -1, 2, 1}, 32768},
                                                         }},
                                                        {10,
                                                         {
                                                             {-18546, {0, 1, -1, 0}, 56936},
                                                             {-3168017, {0, -1, 0, 1}, 4380},
                                                             {-12339, {0, -1, 0, 1}, 3559},
                                                             {2, {0, 1, 2, 1}, 16384},
                                                         }},
                                                        {12,
                                                         {
                                                             {-18546, {0, -1, 1, 0}, 56936},
                                                             {-1584008, {0, 0, -1, 1}, 8759},
                                                             {-6170, {0, 0, -1, 1}, 7117},
                                                             {1, {0, 1, 2, -1}, 32768},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, -1, -2, -1}, {0, 1, 1, 1}},
                                                    {
                                                        {2,
                                                         {
                                                             {-18546, {0, 1, -1, 0}, 56936},
                                                             {1584248, {0, 0, 1, -1}, 8759},
                                                             {6206, {0, 0, 1, -1}, 7117},
                                                             {775, {0, 1, 2, -1}, 32768},
                                                         }},
                                                        {4,
                                                         {
                                                             {-18546, {0, -1, 1, 0}, 56936},
                                                             {239, {0, 1, 0, -1}, 4380},
                                                             {37, {0, 1, 0, -1}, 3559},
                                                             {-772, {0, 1, 2, 1}, 16384},
                                                         }},
                                                        {6,
                                                         {
                                                             {-6170, {0, 2, -1, -1}, 56936},
                                                             {-1584008, {0, -1, 1, 0}, 8759},
                                                             {-6170, {0, -1, 1, 0}, 7117},
                                                             {-773, {0, -1, 2, 1}, 32768},
                                                         }},
                                                        {8,
                                                         {
                                                             {-6170, {0, -2, 1, 1}, 56936},
                                                             {-1584008, {0, 1, -1, 0}, 8759},
                                                             {-6170, {0, 1, -1, 0}, 7117},
                                                             {775, {0, -1, 2, 1}, 32768},
                                                         }},
                                                        {10,
                                                         {
                                                             {-18546, {0, 1, -1, 0}, 56936},
                                                             {239, {0, -1, 0, 1}, 4380},
                                                             {37, {0, -1, 0, 1}, 3559},
                                                             {776, {0, 1, 2, 1}, 16384},
                                                         }},
                                                        {12,
                                                         {
                                                             {-18546, {0, -1, 1, 0}, 56936},
                                                             {1584248, {0, 0, -1, 1}, 8759},
                                                             {6206, {0, 0, -1, 1}, 7117},
                                                             {-773, {0, 1, 2, -1}, 32768},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, -1, -2, -3}, {1, 1, 1, 0}},
                                                    {
                                                        {1,
                                                         {
                                                             {-18546, {1, 1, -2, 0}, 56936},
                                                             {-1584008, {0, 1, -1, 0}, 8759},
                                                             {-6170, {0, 1, -1, 0}, 7117},
                                                             {1, {1, 2, -1, 0}, 32768},
                                                         }},
                                                        {2,
                                                         {
                                                             {-18546, {0, 1, -1, 0}, 56936},
                                                             {-3168017, {1, 0, -1, 0}, 4380},
                                                             {-12339, {1, 0, -1, 0}, 3559},
                                                             {2, {1, 2, 1, 0}, 16384},
                                                         }},
                                                        {3,
                                                         {
                                                             {-18546, {0, 1, -1, 0}, 56936},
                                                             {-1584008, {-1, 1, 0, 0}, 8759},
                                                             {-6170, {-1, 1, 0, 0}, 7117},
                                                             {1, {-1, 2, 1, 0}, 32768},
                                                         }},
                                                        {4,
                                                         {
                                                             {-18546, {0, -1, 1, 0}, 56936},
                                                             {-1584008, {1, -1, 0, 0}, 8759},
                                                             {-6170, {1, -1, 0, 0}, 7117},
                                                             {1, {-1, 2, 1, 0}, 32768},
                                                         }},
                                                        {5,
                                                         {
                                                             {-18546, {0, -1, 1, 0}, 56936},
                                                             {-3168017, {-1, 0, 1, 0}, 4380},
                                                             {-12339, {-1, 0, 1, 0}, 3559},
                                                             {2, {1, 2, 1, 0}, 16384},
                                                         }},
                                                        {6,
                                                         {
                                                             {-18546, {-1, -1, 2, 0}, 56936},
                                                             {-1584008, {0, -1, 1, 0}, 8759},
                                                             {-6170, {0, -1, 1, 0}, 7117},
                                                             {1, {1, 2, -1, 0}, 32768},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, 1, 2, 1}, {0, 1, 1, 1}},
                                                    {
                                                        {2,
                                                         {
                                                             {18582, {0, 1, -1, 0}, 56936},
                                                             {-1584008, {0, 0, 1, -1}, 8759},
                                                             {-6170, {0, 0, 1, -1}, 7117},
                                                             {-773, {0, 1, 2, -1}, 32768},
                                                         }},
                                                        {4,
                                                         {
                                                             {18582, {0, -1, 1, 0}, 56936},
                                                             {239, {0, 1, 0, -1}, 4380},
                                                             {37, {0, 1, 0, -1}, 3559},
                                                             {776, {0, 1, 2, 1}, 16384},
                                                         }},
                                                        {6,
                                                         {
                                                             {6206, {0, 2, -1, -1}, 56936},
                                                             {1584248, {0, -1, 1, 0}, 8759},
                                                             {6206, {0, -1, 1, 0}, 7117},
                                                             {775, {0, -1, 2, 1}, 32768},
                                                         }},
                                                        {8,
                                                         {
                                                             {6206, {0, -2, 1, 1}, 56936},
                                                             {1584248, {0, 1, -1, 0}, 8759},
                                                             {6206, {0, 1, -1, 0}, 7117},
                                                             {-773, {0, -1, 2, 1}, 32768},
                                                         }},
                                                        {10,
                                                         {
                                                             {18582, {0, 1, -1, 0}, 56936},
                                                             {239, {0, -1, 0, 1}, 4380},
                                                             {37, {0, -1, 0, 1}, 3559},
                                                             {-772, {0, 1, 2, 1}, 16384},
                                                         }},
                                                        {12,
                                                         {
                                                             {18582, {0, -1, 1, 0}, 56936},
                                                             {-1584008, {0, 0, -1, 1}, 8759},
                                                             {-6170, {0, 0, -1, 1}, 7117},
                                                             {775, {0, 1, 2, -1}, 32768},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, -1, -2, -1}, {1, 1, 1, 1}},
                                                    {
                                                        {4,
                                                         {
                                                             {-222510, {-6, -5, 14, -3}, 4067},
                                                             {-6334836, {4, 1, 0, -5}, 626},
                                                             {-24494, {4, 1, 0, -5}, 508},
                                                             {-3087, {1, 2, 7, 4}, 4681},
                                                         }},
                                                        {6,
                                                         {
                                                             {-24715, {-1, 1, 1, -1}, 28468},
                                                             {-6335315, {3, -1, 1, -3}, 876},
                                                             {-24568, {3, -1, 1, -3}, 712},
                                                             {-772, {1, 1, 1, 1}, 16384},
                                                         }},
                                                        {7,
                                                         {
                                                             {-37018, {5, 2, -1, -6}, 9489},
                                                             {-3168017, {-1, 0, 1, 0}, 4380},
                                                             {-12339, {-1, 0, 1, 0}, 3559},
                                                             {-2318, {-2, 1, 4, 3}, 10923},
                                                         }},
                                                        {0,
                                                         {
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                         }},
                                                        {0,
                                                         {
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                         }},
                                                        {0,
                                                         {
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, 1, 2, 3}, {0, 1, 1, 1}},
                                                    {
                                                        {2,
                                                         {
                                                             {18582, {0, 1, -1, 0}, 56936},
                                                             {1584248, {0, 0, 1, -1}, 8759},
                                                             {6206, {0, 0, 1, -1}, 7117},
                                                             {1, {0, 1, 2, -1}, 32768},
                                                         }},
                                                        {4,
                                                         {
                                                             {18582, {0, -1, 1, 0}, 56936},
                                                             {3168495, {0, 1, 0, -1}, 4380},
                                                             {12413, {0, 1, 0, -1}, 3559},
                                                             {2, {0, 1, 2, 1}, 16384},
                                                         }},
                                                        {6,
                                                         {
                                                             {18582, {0, 2, -1, -1}, 56936},
                                                             {1584248, {0, -1, 1, 0}, 8759},
                                                             {6206, {0, -1, 1, 0}, 7117},
                                                             {1, {0, -1, 2, 1}, 32768},
                                                         }},
                                                        {8,
                                                         {
                                                             {18582, {0, -2, 1, 1}, 56936},
                                                             {1584248, {0, 1, -1, 0}, 8759},
                                                             {6206, {0, 1, -1, 0}, 7117},
                                                             {1, {0, -1, 2, 1}, 32768},
                                                         }},
                                                        {10,
                                                         {
                                                             {18582, {0, 1, -1, 0}, 56936},
                                                             {3168495, {0, -1, 0, 1}, 4380},
                                                             {12413, {0, -1, 0, 1}, 3559},
                                                             {2, {0, 1, 2, 1}, 16384},
                                                         }},
                                                        {12,
                                                         {
                                                             {18582, {0, -1, 1, 0}, 56936},
                                                             {1584248, {0, 0, -1, 1}, 8759},
                                                             {6206, {0, 0, -1, 1}, 7117},
                                                             {1, {0, 1, 2, -1}, 32768},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, 1, 0, -1}, {1, 1, 1, 1}},
                                                    {
                                                        {1,
                                                         {
                                                             {-37018, {6, 1, -2, -5}, 9489},
                                                             {-3168017, {0, 1, 0, -1}, 4380},
                                                             {-12339, {0, 1, 0, -1}, 3559},
                                                             {-2318, {3, 4, 1, -2}, 10923},
                                                         }},
                                                        {9,
                                                         {
                                                             {37, {1, -1, -1, 1}, 28468},
                                                             {-6335315, {-3, 1, -1, 3}, 876},
                                                             {-24568, {-3, 1, -1, 3}, 712},
                                                             {-772, {1, 1, 1, 1}, 16384},
                                                         }},
                                                        {13,
                                                         {
                                                             {49762, {3, -14, 5, 6}, 4067},
                                                             {-6334836, {-5, 0, 1, 4}, 626},
                                                             {-24494, {-5, 0, 1, 4}, 508},
                                                             {-3087, {4, 7, 2, 1}, 4681},
                                                         }},
                                                        {0,
                                                         {
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                         }},
                                                        {0,
                                                         {
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                         }},
                                                        {0,
                                                         {
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, -1, -2, -1}, {1, 1, 0, 1}},
                                                    {
                                                        {1,
                                                         {
                                                             {-6170, {1, 0, 0, -1}, 56936},
                                                             {239, {0, 1, 0, -1}, 4380},
                                                             {37, {0, 1, 0, -1}, 3559},
                                                             {776, {2, 3, 0, -1}, 16384},
                                                         }},
                                                        {2,
                                                         {
                                                             {-30885, {-1, 3, 0, -2}, 18979},
                                                             {-1583769, {1, 0, 0, -1}, 2920},
                                                             {-6133, {1, 0, 0, -1}, 2372},
                                                             {-771, {2, 3, 0, 1}, 10923},
                                                         }},
                                                        {3,
                                                         {
                                                             {-6170, {1, 0, 0, -1}, 56936},
                                                             {-1584008, {-1, 1, 0, 0}, 8759},
                                                             {-6170, {-1, 1, 0, 0}, 7117},
                                                             {-773, {-2, 3, 0, 1}, 32768},
                                                         }},
                                                        {8,
                                                         {
                                                             {-6170, {-1, 0, 0, 1}, 56936},
                                                             {-1584008, {1, -1, 0, 0}, 8759},
                                                             {-6170, {1, -1, 0, 0}, 7117},
                                                             {775, {-2, 3, 0, 1}, 32768},
                                                         }},
                                                        {9,
                                                         {
                                                             {-30885, {1, -3, 0, 2}, 18979},
                                                             {-1583769, {-1, 0, 0, 1}, 2920},
                                                             {-6133, {-1, 0, 0, 1}, 2372},
                                                             {777, {2, 3, 0, 1}, 10923},
                                                         }},
                                                        {10,
                                                         {
                                                             {-6170, {-1, 0, 0, 1}, 56936},
                                                             {239, {0, -1, 0, 1}, 4380},
                                                             {37, {0, -1, 0, 1}, 3559},
                                                             {-772, {2, 3, 0, -1}, 16384},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, 1, 2, 3}, {1, 1, 0, 1}},
                                                    {
                                                        {1,
                                                         {
                                                             {18582, {1, 0, 0, -1}, 56936},
                                                             {3168495, {0, 1, 0, -1}, 4380},
                                                             {12413, {0, 1, 0, -1}, 3559},
                                                             {2, {2, 3, 0, -1}, 16384},
                                                         }},
                                                        {2,
                                                         {
                                                             {55747, {-1, 3, 0, -2}, 18979},
                                                             {4752743, {1, 0, 0, -1}, 2920},
                                                             {18619, {1, 0, 0, -1}, 2372},
                                                             {3, {2, 3, 0, 1}, 10923},
                                                         }},
                                                        {3,
                                                         {
                                                             {18582, {1, 0, 0, -1}, 56936},
                                                             {1584248, {-1, 1, 0, 0}, 8759},
                                                             {6206, {-1, 1, 0, 0}, 7117},
                                                             {1, {-2, 3, 0, 1}, 32768},
                                                         }},
                                                        {8,
                                                         {
                                                             {18582, {-1, 0, 0, 1}, 56936},
                                                             {1584248, {1, -1, 0, 0}, 8759},
                                                             {6206, {1, -1, 0, 0}, 7117},
                                                             {1, {-2, 3, 0, 1}, 32768},
                                                         }},
                                                        {9,
                                                         {
                                                             {55747, {1, -3, 0, 2}, 18979},
                                                             {4752743, {-1, 0, 0, 1}, 2920},
                                                             {18619, {-1, 0, 0, 1}, 2372},
                                                             {3, {2, 3, 0, 1}, 10923},
                                                         }},
                                                        {10,
                                                         {
                                                             {18582, {-1, 0, 0, 1}, 56936},
                                                             {3168495, {0, -1, 0, 1}, 4380},
                                                             {12413, {0, -1, 0, 1}, 3559},
                                                             {2, {2, 3, 0, -1}, 16384},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, -1, 0, 1}, {1, 1, 1, 0}},
                                                    {
                                                        {1,
                                                         {
                                                             {6206, {1, 1, -2, 0}, 56936},
                                                             {1584248, {0, 1, -1, 0}, 8759},
                                                             {6206, {0, 1, -1, 0}, 7117},
                                                             {775, {1, 2, -1, 0}, 32768},
                                                         }},
                                                        {2,
                                                         {
                                                             {-6170, {0, 1, -1, 0}, 56936},
                                                             {239, {1, 0, -1, 0}, 4380},
                                                             {37, {1, 0, -1, 0}, 3559},
                                                             {-772, {1, 2, 1, 0}, 16384},
                                                         }},
                                                        {3,
                                                         {
                                                             {-6170, {0, 1, -1, 0}, 56936},
                                                             {-1584008, {-1, 1, 0, 0}, 8759},
                                                             {-6170, {-1, 1, 0, 0}, 7117},
                                                             {-773, {-1, 2, 1, 0}, 32768},
                                                         }},
                                                        {4,
                                                         {
                                                             {-6170, {0, -1, 1, 0}, 56936},
                                                             {-1584008, {1, -1, 0, 0}, 8759},
                                                             {-6170, {1, -1, 0, 0}, 7117},
                                                             {775, {-1, 2, 1, 0}, 32768},
                                                         }},
                                                        {5,
                                                         {
                                                             {-6170, {0, -1, 1, 0}, 56936},
                                                             {239, {-1, 0, 1, 0}, 4380},
                                                             {37, {-1, 0, 1, 0}, 3559},
                                                             {776, {1, 2, 1, 0}, 16384},
                                                         }},
                                                        {6,
                                                         {
                                                             {6206, {-1, -1, 2, 0}, 56936},
                                                             {1584248, {0, -1, 1, 0}, 8759},
                                                             {6206, {0, -1, 1, 0}, 7117},
                                                             {-773, {1, 2, -1, 0}, 32768},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, 1, 0, -1}, {0, 1, 1, 1}},
                                                    {
                                                        {2,
                                                         {
                                                             {6206, {0, 1, -1, 0}, 56936},
                                                             {-1584008, {0, 0, 1, -1}, 8759},
                                                             {-6170, {0, 0, 1, -1}, 7117},
                                                             {1, {0, 1, 2, -1}, 32768},
                                                         }},
                                                        {4,
                                                         {
                                                             {6206, {0, -1, 1, 0}, 56936},
                                                             {-3168017, {0, 1, 0, -1}, 4380},
                                                             {-12339, {0, 1, 0, -1}, 3559},
                                                             {2, {0, 1, 2, 1}, 16384},
                                                         }},
                                                        {6,
                                                         {
                                                             {6206, {0, 2, -1, -1}, 56936},
                                                             {-1584008, {0, -1, 1, 0}, 8759},
                                                             {-6170, {0, -1, 1, 0}, 7117},
                                                             {1, {0, -1, 2, 1}, 32768},
                                                         }},
                                                        {8,
                                                         {
                                                             {6206, {0, -2, 1, 1}, 56936},
                                                             {-1584008, {0, 1, -1, 0}, 8759},
                                                             {-6170, {0, 1, -1, 0}, 7117},
                                                             {1, {0, -1, 2, 1}, 32768},
                                                         }},
                                                        {10,
                                                         {
                                                             {6206, {0, 1, -1, 0}, 56936},
                                                             {-3168017, {0, -1, 0, 1}, 4380},
                                                             {-12339, {0, -1, 0, 1}, 3559},
                                                             {2, {0, 1, 2, 1}, 16384},
                                                         }},
                                                        {12,
                                                         {
                                                             {6206, {0, -1, 1, 0}, 56936},
                                                             {-1584008, {0, 0, -1, 1}, 8759},
                                                             {-6170, {0, 0, -1, 1}, 7117},
                                                             {1, {0, 1, 2, -1}, 32768},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, -1, 0, -1}, {1, 1, 1, 1}},
                                                    {
                                                        {2,
                                                         {
                                                             {-123502, {-3, 14, -5, -6}, 4067},
                                                             {-6334836, {5, 0, -1, -4}, 626},
                                                             {-24494, {5, 0, -1, -4}, 508},
                                                             {-2314, {4, 7, 2, 1}, 4681},
                                                         }},
                                                        {10,
                                                         {
                                                             {-12339, {-1, 1, -1, 1}, 28468},
                                                             {479, {1, -1, -1, 1}, 2190},
                                                             {74, {1, -1, -1, 1}, 1779},
                                                             {-1543, {1, 3, 3, 1}, 8192},
                                                         }},
                                                        {3,
                                                         {
                                                             {-12339, {1, 1, -1, -1}, 28468},
                                                             {-3168017, {-1, 1, 1, -1}, 4380},
                                                             {-12339, {-1, 1, 1, -1}, 3559},
                                                             {-1545, {-1, 3, 3, -1}, 16384},
                                                         }},
                                                        {11,
                                                         {
                                                             {-49246, {6, 5, -14, 3}, 4067},
                                                             {-6334836, {-4, -1, 0, 5}, 626},
                                                             {-24494, {-4, -1, 0, 5}, 508},
                                                             {-2314, {1, 2, 7, 4}, 4681},
                                                         }},
                                                        {0,
                                                         {
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                         }},
                                                        {0,
                                                         {
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, -1, 0, -1}, {0, 1, 1, 1}},
                                                    {
                                                        {2,
                                                         {
                                                             {-6170, {0, 1, -1, 0}, 56936},
                                                             {-1584008, {0, 0, 1, -1}, 8759},
                                                             {-6170, {0, 0, 1, -1}, 7117},
                                                             {-773, {0, 1, 2, -1}, 32768},
                                                         }},
                                                        {4,
                                                         {
                                                             {-6170, {0, -1, 1, 0}, 56936},
                                                             {239, {0, 1, 0, -1}, 4380},
                                                             {37, {0, 1, 0, -1}, 3559},
                                                             {776, {0, 1, 2, 1}, 16384},
                                                         }},
                                                        {6,
                                                         {
                                                             {-18546, {0, 2, -1, -1}, 56936},
                                                             {1584248, {0, -1, 1, 0}, 8759},
                                                             {6206, {0, -1, 1, 0}, 7117},
                                                             {775, {0, -1, 2, 1}, 32768},
                                                         }},
                                                        {8,
                                                         {
                                                             {-18546, {0, -2, 1, 1}, 56936},
                                                             {1584248, {0, 1, -1, 0}, 8759},
                                                             {6206, {0, 1, -1, 0}, 7117},
                                                             {-773, {0, -1, 2, 1}, 32768},
                                                         }},
                                                        {10,
                                                         {
                                                             {-6170, {0, 1, -1, 0}, 56936},
                                                             {239, {0, -1, 0, 1}, 4380},
                                                             {37, {0, -1, 0, 1}, 3559},
                                                             {-772, {0, 1, 2, 1}, 16384},
                                                         }},
                                                        {12,
                                                         {
                                                             {-6170, {0, -1, 1, 0}, 56936},
                                                             {-1584008, {0, 0, -1, 1}, 8759},
                                                             {-6170, {0, 0, -1, 1}, 7117},
                                                             {775, {0, 1, 2, -1}, 32768},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, 1, 2, 3}, {1, 1, 1, 1}},
                                                    {
                                                        {8,
                                                         {
                                                             {111495, {-5, -2, 1, 6}, 9489},
                                                             {3168495, {1, 0, -1, 0}, 4380},
                                                             {12413, {1, 0, -1, 0}, 3559},
                                                             {3, {-2, 1, 4, 3}, 10923},
                                                         }},
                                                        {12,
                                                         {
                                                             {37165, {-1, -1, 1, 1}, 28468},
                                                             {3168495, {1, -1, -1, 1}, 4380},
                                                             {12413, {1, -1, -1, 1}, 3559},
                                                             {2, {-1, 3, 3, -1}, 16384},
                                                         }},
                                                        {14,
                                                         {
                                                             {111495, {-6, -1, 2, 5}, 9489},
                                                             {3168495, {0, -1, 0, 1}, 4380},
                                                             {12413, {0, -1, 0, 1}, 3559},
                                                             {3, {3, 4, 1, -2}, 10923},
                                                         }},
                                                        {1,
                                                         {
                                                             {111495, {6, 1, -2, -5}, 9489},
                                                             {3168495, {0, 1, 0, -1}, 4380},
                                                             {12413, {0, 1, 0, -1}, 3559},
                                                             {3, {3, 4, 1, -2}, 10923},
                                                         }},
                                                        {3,
                                                         {
                                                             {37165, {1, 1, -1, -1}, 28468},
                                                             {3168495, {-1, 1, 1, -1}, 4380},
                                                             {12413, {-1, 1, 1, -1}, 3559},
                                                             {2, {-1, 3, 3, -1}, 16384},
                                                         }},
                                                        {7,
                                                         {
                                                             {111495, {5, 2, -1, -6}, 9489},
                                                             {3168495, {-1, 0, 1, 0}, 4380},
                                                             {12413, {-1, 0, 1, 0}, 3559},
                                                             {3, {-2, 1, 4, 3}, 10923},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, 1, 0, 1}, {1, 0, 1, 1}},
                                                    {
                                                        {1,
                                                         {
                                                             {6206, {1, 0, 0, -1}, 56936},
                                                             {1584248, {0, 0, 1, -1}, 8759},
                                                             {6206, {0, 0, 1, -1}, 7117},
                                                             {775, {1, 0, 3, -2}, 32768},
                                                         }},
                                                        {4,
                                                         {
                                                             {6243, {-2, 0, 3, -1}, 18979},
                                                             {1584487, {1, 0, 0, -1}, 2920},
                                                             {6243, {1, 0, 0, -1}, 2372},
                                                             {-771, {1, 0, 3, 2}, 10923},
                                                         }},
                                                        {5,
                                                         {
                                                             {6206, {1, 0, 0, -1}, 56936},
                                                             {239, {-1, 0, 1, 0}, 4380},
                                                             {37, {-1, 0, 1, 0}, 3559},
                                                             {-772, {-1, 0, 3, 2}, 16384},
                                                         }},
                                                        {8,
                                                         {
                                                             {6206, {-1, 0, 0, 1}, 56936},
                                                             {239, {1, 0, -1, 0}, 4380},
                                                             {37, {1, 0, -1, 0}, 3559},
                                                             {776, {-1, 0, 3, 2}, 16384},
                                                         }},
                                                        {9,
                                                         {
                                                             {6243, {2, 0, -3, 1}, 18979},
                                                             {1584487, {-1, 0, 0, 1}, 2920},
                                                             {6243, {-1, 0, 0, 1}, 2372},
                                                             {777, {1, 0, 3, 2}, 10923},
                                                         }},
                                                        {12,
                                                         {
                                                             {6206, {-1, 0, 0, 1}, 56936},
                                                             {1584248, {0, 0, -1, 1}, 8759},
                                                             {6206, {0, 0, -1, 1}, 7117},
                                                             {-773, {1, 0, 3, -2}, 32768},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, 1, 2, 1}, {1, 1, 0, 1}},
                                                    {
                                                        {1,
                                                         {
                                                             {6206, {1, 0, 0, -1}, 56936},
                                                             {239, {0, 1, 0, -1}, 4380},
                                                             {37, {0, 1, 0, -1}, 3559},
                                                             {-772, {2, 3, 0, -1}, 16384},
                                                         }},
                                                        {2,
                                                         {
                                                             {30995, {-1, 3, 0, -2}, 18979},
                                                             {1584487, {1, 0, 0, -1}, 2920},
                                                             {6243, {1, 0, 0, -1}, 2372},
                                                             {777, {2, 3, 0, 1}, 10923},
                                                         }},
                                                        {3,
                                                         {
                                                             {6206, {1, 0, 0, -1}, 56936},
                                                             {1584248, {-1, 1, 0, 0}, 8759},
                                                             {6206, {-1, 1, 0, 0}, 7117},
                                                             {775, {-2, 3, 0, 1}, 32768},
                                                         }},
                                                        {8,
                                                         {
                                                             {6206, {-1, 0, 0, 1}, 56936},
                                                             {1584248, {1, -1, 0, 0}, 8759},
                                                             {6206, {1, -1, 0, 0}, 7117},
                                                             {-773, {-2, 3, 0, 1}, 32768},
                                                         }},
                                                        {9,
                                                         {
                                                             {30995, {1, -3, 0, 2}, 18979},
                                                             {1584487, {-1, 0, 0, 1}, 2920},
                                                             {6243, {-1, 0, 0, 1}, 2372},
                                                             {-771, {2, 3, 0, 1}, 10923},
                                                         }},
                                                        {10,
                                                         {
                                                             {6206, {-1, 0, 0, 1}, 56936},
                                                             {239, {0, -1, 0, 1}, 4380},
                                                             {37, {0, -1, 0, 1}, 3559},
                                                             {776, {2, 3, 0, -1}, 16384},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, -1, 0, -1}, {1, 1, 1, 0}},
                                                    {
                                                        {1,
                                                         {
                                                             {6206, {1, 1, -2, 0}, 56936},
                                                             {1584248, {0, 1, -1, 0}, 8759},
                                                             {6206, {0, 1, -1, 0}, 7117},
                                                             {775, {1, 2, -1, 0}, 32768},
                                                         }},
                                                        {2,
                                                         {
                                                             {-6170, {0, 1, -1, 0}, 56936},
                                                             {239, {1, 0, -1, 0}, 4380},
                                                             {37, {1, 0, -1, 0}, 3559},
                                                             {-772, {1, 2, 1, 0}, 16384},
                                                         }},
                                                        {3,
                                                         {
                                                             {-6170, {0, 1, -1, 0}, 56936},
                                                             {-1584008, {-1, 1, 0, 0}, 8759},
                                                             {-6170, {-1, 1, 0, 0}, 7117},
                                                             {-773, {-1, 2, 1, 0}, 32768},
                                                         }},
                                                        {4,
                                                         {
                                                             {-6170, {0, -1, 1, 0}, 56936},
                                                             {-1584008, {1, -1, 0, 0}, 8759},
                                                             {-6170, {1, -1, 0, 0}, 7117},
                                                             {775, {-1, 2, 1, 0}, 32768},
                                                         }},
                                                        {5,
                                                         {
                                                             {-6170, {0, -1, 1, 0}, 56936},
                                                             {239, {-1, 0, 1, 0}, 4380},
                                                             {37, {-1, 0, 1, 0}, 3559},
                                                             {776, {1, 2, 1, 0}, 16384},
                                                         }},
                                                        {6,
                                                         {
                                                             {6206, {-1, -1, 2, 0}, 56936},
                                                             {1584248, {0, -1, 1, 0}, 8759},
                                                             {6206, {0, -1, 1, 0}, 7117},
                                                             {-773, {1, 2, -1, 0}, 32768},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, 1, 2, 1}, {1, 0, 1, 1}},
                                                    {
                                                        {1,
                                                         {
                                                             {6206, {1, 0, 0, -1}, 56936},
                                                             {-1584008, {0, 0, 1, -1}, 8759},
                                                             {-6170, {0, 0, 1, -1}, 7117},
                                                             {-1546, {1, 0, 3, -2}, 32768},
                                                         }},
                                                        {4,
                                                         {
                                                             {43371, {-2, 0, 3, -1}, 18979},
                                                             {1584487, {1, 0, 0, -1}, 2920},
                                                             {6243, {1, 0, 0, -1}, 2372},
                                                             {1550, {1, 0, 3, 2}, 10923},
                                                         }},
                                                        {5,
                                                         {
                                                             {6206, {1, 0, 0, -1}, 56936},
                                                             {3168495, {-1, 0, 1, 0}, 4380},
                                                             {12413, {-1, 0, 1, 0}, 3559},
                                                             {1549, {-1, 0, 3, 2}, 16384},
                                                         }},
                                                        {8,
                                                         {
                                                             {6206, {-1, 0, 0, 1}, 56936},
                                                             {3168495, {1, 0, -1, 0}, 4380},
                                                             {12413, {1, 0, -1, 0}, 3559},
                                                             {-1545, {-1, 0, 3, 2}, 16384},
                                                         }},
                                                        {9,
                                                         {
                                                             {43371, {2, 0, -3, 1}, 18979},
                                                             {1584487, {-1, 0, 0, 1}, 2920},
                                                             {6243, {-1, 0, 0, 1}, 2372},
                                                             {-1544, {1, 0, 3, 2}, 10923},
                                                         }},
                                                        {12,
                                                         {
                                                             {6206, {-1, 0, 0, 1}, 56936},
                                                             {-1584008, {0, 0, -1, 1}, 8759},
                                                             {-6170, {0, 0, -1, 1}, 7117},
                                                             {1548, {1, 0, 3, -2}, 32768},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, 1, 0, -1}, {1, 1, 1, 0}},
                                                    {
                                                        {1,
                                                         {
                                                             {-6170, {1, 1, -2, 0}, 56936},
                                                             {-1584008, {0, 1, -1, 0}, 8759},
                                                             {-6170, {0, 1, -1, 0}, 7117},
                                                             {-773, {1, 2, -1, 0}, 32768},
                                                         }},
                                                        {2,
                                                         {
                                                             {6206, {0, 1, -1, 0}, 56936},
                                                             {239, {1, 0, -1, 0}, 4380},
                                                             {37, {1, 0, -1, 0}, 3559},
                                                             {776, {1, 2, 1, 0}, 16384},
                                                         }},
                                                        {3,
                                                         {
                                                             {6206, {0, 1, -1, 0}, 56936},
                                                             {1584248, {-1, 1, 0, 0}, 8759},
                                                             {6206, {-1, 1, 0, 0}, 7117},
                                                             {775, {-1, 2, 1, 0}, 32768},
                                                         }},
                                                        {4,
                                                         {
                                                             {6206, {0, -1, 1, 0}, 56936},
                                                             {1584248, {1, -1, 0, 0}, 8759},
                                                             {6206, {1, -1, 0, 0}, 7117},
                                                             {-773, {-1, 2, 1, 0}, 32768},
                                                         }},
                                                        {5,
                                                         {
                                                             {6206, {0, -1, 1, 0}, 56936},
                                                             {239, {-1, 0, 1, 0}, 4380},
                                                             {37, {-1, 0, 1, 0}, 3559},
                                                             {-772, {1, 2, 1, 0}, 16384},
                                                         }},
                                                        {6,
                                                         {
                                                             {-6170, {-1, -1, 2, 0}, 56936},
                                                             {-1584008, {0, -1, 1, 0}, 8759},
                                                             {-6170, {0, -1, 1, 0}, 7117},
                                                             {775, {1, 2, -1, 0}, 32768},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, 1, 2, 3}, {1, 1, 1, 0}},
                                                    {
                                                        {1,
                                                         {
                                                             {18582, {1, 1, -2, 0}, 56936},
                                                             {1584248, {0, 1, -1, 0}, 8759},
                                                             {6206, {0, 1, -1, 0}, 7117},
                                                             {1, {1, 2, -1, 0}, 32768},
                                                         }},
                                                        {2,
                                                         {
                                                             {18582, {0, 1, -1, 0}, 56936},
                                                             {3168495, {1, 0, -1, 0}, 4380},
                                                             {12413, {1, 0, -1, 0}, 3559},
                                                             {2, {1, 2, 1, 0}, 16384},
                                                         }},
                                                        {3,
                                                         {
                                                             {18582, {0, 1, -1, 0}, 56936},
                                                             {1584248, {-1, 1, 0, 0}, 8759},
                                                             {6206, {-1, 1, 0, 0}, 7117},
                                                             {1, {-1, 2, 1, 0}, 32768},
                                                         }},
                                                        {4,
                                                         {
                                                             {18582, {0, -1, 1, 0}, 56936},
                                                             {1584248, {1, -1, 0, 0}, 8759},
                                                             {6206, {1, -1, 0, 0}, 7117},
                                                             {1, {-1, 2, 1, 0}, 32768},
                                                         }},
                                                        {5,
                                                         {
                                                             {18582, {0, -1, 1, 0}, 56936},
                                                             {3168495, {-1, 0, 1, 0}, 4380},
                                                             {12413, {-1, 0, 1, 0}, 3559},
                                                             {2, {1, 2, 1, 0}, 16384},
                                                         }},
                                                        {6,
                                                         {
                                                             {18582, {-1, -1, 2, 0}, 56936},
                                                             {1584248, {0, -1, 1, 0}, 8759},
                                                             {6206, {0, -1, 1, 0}, 7117},
                                                             {1, {1, 2, -1, 0}, 32768},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, 1, 0, 1}, {1, 1, 1, 0}},
                                                    {
                                                        {1,
                                                         {
                                                             {-6170, {1, 1, -2, 0}, 56936},
                                                             {-1584008, {0, 1, -1, 0}, 8759},
                                                             {-6170, {0, 1, -1, 0}, 7117},
                                                             {-773, {1, 2, -1, 0}, 32768},
                                                         }},
                                                        {2,
                                                         {
                                                             {6206, {0, 1, -1, 0}, 56936},
                                                             {239, {1, 0, -1, 0}, 4380},
                                                             {37, {1, 0, -1, 0}, 3559},
                                                             {776, {1, 2, 1, 0}, 16384},
                                                         }},
                                                        {3,
                                                         {
                                                             {6206, {0, 1, -1, 0}, 56936},
                                                             {1584248, {-1, 1, 0, 0}, 8759},
                                                             {6206, {-1, 1, 0, 0}, 7117},
                                                             {775, {-1, 2, 1, 0}, 32768},
                                                         }},
                                                        {4,
                                                         {
                                                             {6206, {0, -1, 1, 0}, 56936},
                                                             {1584248, {1, -1, 0, 0}, 8759},
                                                             {6206, {1, -1, 0, 0}, 7117},
                                                             {-773, {-1, 2, 1, 0}, 32768},
                                                         }},
                                                        {5,
                                                         {
                                                             {6206, {0, -1, 1, 0}, 56936},
                                                             {239, {-1, 0, 1, 0}, 4380},
                                                             {37, {-1, 0, 1, 0}, 3559},
                                                             {-772, {1, 2, 1, 0}, 16384},
                                                         }},
                                                        {6,
                                                         {
                                                             {-6170, {-1, -1, 2, 0}, 56936},
                                                             {-1584008, {0, -1, 1, 0}, 8759},
                                                             {-6170, {0, -1, 1, 0}, 7117},
                                                             {775, {1, 2, -1, 0}, 32768},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, -1, 0, 1}, {1, 1, 1, 1}},
                                                    {
                                                        {2,
                                                         {
                                                             {-49246, {-3, 14, -5, -6}, 4067},
                                                             {6338188, {5, 0, -1, -4}, 626},
                                                             {25010, {5, 0, -1, -4}, 508},
                                                             {-3087, {4, 7, 2, 1}, 4681},
                                                         }},
                                                        {6,
                                                         {
                                                             {37, {-1, 1, 1, -1}, 28468},
                                                             {6337709, {3, -1, 1, -3}, 876},
                                                             {24936, {3, -1, 1, -3}, 712},
                                                             {-772, {1, 1, 1, 1}, 16384},
                                                         }},
                                                        {14,
                                                         {
                                                             {37239, {-6, -1, 2, 5}, 9489},
                                                             {3168495, {0, -1, 0, 1}, 4380},
                                                             {12413, {0, -1, 0, 1}, 3559},
                                                             {-2318, {3, 4, 1, -2}, 10923},
                                                         }},
                                                        {0,
                                                         {
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                         }},
                                                        {0,
                                                         {
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                         }},
                                                        {0,
                                                         {
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, -1, -2, -3}, {1, 0, 1, 1}},
                                                    {
                                                        {1,
                                                         {
                                                             {-18546, {1, 0, 0, -1}, 56936},
                                                             {-1584008, {0, 0, 1, -1}, 8759},
                                                             {-6170, {0, 0, 1, -1}, 7117},
                                                             {1, {1, 0, 3, -2}, 32768},
                                                         }},
                                                        {4,
                                                         {
                                                             {-55637, {-2, 0, 3, -1}, 18979},
                                                             {-4752025, {1, 0, 0, -1}, 2920},
                                                             {-18509, {1, 0, 0, -1}, 2372},
                                                             {3, {1, 0, 3, 2}, 10923},
                                                         }},
                                                        {5,
                                                         {
                                                             {-18546, {1, 0, 0, -1}, 56936},
                                                             {-3168017, {-1, 0, 1, 0}, 4380},
                                                             {-12339, {-1, 0, 1, 0}, 3559},
                                                             {2, {-1, 0, 3, 2}, 16384},
                                                         }},
                                                        {8,
                                                         {
                                                             {-18546, {-1, 0, 0, 1}, 56936},
                                                             {-3168017, {1, 0, -1, 0}, 4380},
                                                             {-12339, {1, 0, -1, 0}, 3559},
                                                             {2, {-1, 0, 3, 2}, 16384},
                                                         }},
                                                        {9,
                                                         {
                                                             {-55637, {2, 0, -3, 1}, 18979},
                                                             {-4752025, {-1, 0, 0, 1}, 2920},
                                                             {-18509, {-1, 0, 0, 1}, 2372},
                                                             {3, {1, 0, 3, 2}, 10923},
                                                         }},
                                                        {12,
                                                         {
                                                             {-18546, {-1, 0, 0, 1}, 56936},
                                                             {-1584008, {0, 0, -1, 1}, 8759},
                                                             {-6170, {0, 0, -1, 1}, 7117},
                                                             {1, {1, 0, 3, -2}, 32768},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, 1, 0, 1}, {1, 1, 1, 1}},
                                                    {
                                                        {4,
                                                         {
                                                             {49762, {-6, -5, 14, -3}, 4067},
                                                             {6338188, {4, 1, 0, -5}, 626},
                                                             {25010, {4, 1, 0, -5}, 508},
                                                             {-2314, {1, 2, 7, 4}, 4681},
                                                         }},
                                                        {12,
                                                         {
                                                             {12413, {-1, -1, 1, 1}, 28468},
                                                             {3168495, {1, -1, -1, 1}, 4380},
                                                             {12413, {1, -1, -1, 1}, 3559},
                                                             {-1545, {-1, 3, 3, -1}, 16384},
                                                         }},
                                                        {5,
                                                         {
                                                             {12413, {1, -1, 1, -1}, 28468},
                                                             {479, {-1, 1, 1, -1}, 2190},
                                                             {74, {-1, 1, 1, -1}, 1779},
                                                             {-1543, {1, 3, 3, 1}, 8192},
                                                         }},
                                                        {13,
                                                         {
                                                             {124018, {3, -14, 5, 6}, 4067},
                                                             {6338188, {-5, 0, 1, 4}, 626},
                                                             {25010, {-5, 0, 1, 4}, 508},
                                                             {-2314, {4, 7, 2, 1}, 4681},
                                                         }},
                                                        {0,
                                                         {
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                         }},
                                                        {0,
                                                         {
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, -1, 0, 1}, {1, 0, 1, 1}},
                                                    {
                                                        {1,
                                                         {
                                                             {6206, {1, 0, 0, -1}, 56936},
                                                             {1584248, {0, 0, 1, -1}, 8759},
                                                             {6206, {0, 0, 1, -1}, 7117},
                                                             {775, {1, 0, 3, -2}, 32768},
                                                         }},
                                                        {4,
                                                         {
                                                             {6243, {-2, 0, 3, -1}, 18979},
                                                             {1584487, {1, 0, 0, -1}, 2920},
                                                             {6243, {1, 0, 0, -1}, 2372},
                                                             {-771, {1, 0, 3, 2}, 10923},
                                                         }},
                                                        {5,
                                                         {
                                                             {6206, {1, 0, 0, -1}, 56936},
                                                             {239, {-1, 0, 1, 0}, 4380},
                                                             {37, {-1, 0, 1, 0}, 3559},
                                                             {-772, {-1, 0, 3, 2}, 16384},
                                                         }},
                                                        {8,
                                                         {
                                                             {6206, {-1, 0, 0, 1}, 56936},
                                                             {239, {1, 0, -1, 0}, 4380},
                                                             {37, {1, 0, -1, 0}, 3559},
                                                             {776, {-1, 0, 3, 2}, 16384},
                                                         }},
                                                        {9,
                                                         {
                                                             {6243, {2, 0, -3, 1}, 18979},
                                                             {1584487, {-1, 0, 0, 1}, 2920},
                                                             {6243, {-1, 0, 0, 1}, 2372},
                                                             {777, {1, 0, 3, 2}, 10923},
                                                         }},
                                                        {12,
                                                         {
                                                             {6206, {-1, 0, 0, 1}, 56936},
                                                             {1584248, {0, 0, -1, 1}, 8759},
                                                             {6206, {0, 0, -1, 1}, 7117},
                                                             {-773, {1, 0, 3, -2}, 32768},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, -1, 0, -1}, {1, 0, 1, 1}},
                                                    {
                                                        {1,
                                                         {
                                                             {-6170, {1, 0, 0, -1}, 56936},
                                                             {-1584008, {0, 0, 1, -1}, 8759},
                                                             {-6170, {0, 0, 1, -1}, 7117},
                                                             {-773, {1, 0, 3, -2}, 32768},
                                                         }},
                                                        {4,
                                                         {
                                                             {-6133, {-2, 0, 3, -1}, 18979},
                                                             {-1583769, {1, 0, 0, -1}, 2920},
                                                             {-6133, {1, 0, 0, -1}, 2372},
                                                             {777, {1, 0, 3, 2}, 10923},
                                                         }},
                                                        {5,
                                                         {
                                                             {-6170, {1, 0, 0, -1}, 56936},
                                                             {239, {-1, 0, 1, 0}, 4380},
                                                             {37, {-1, 0, 1, 0}, 3559},
                                                             {776, {-1, 0, 3, 2}, 16384},
                                                         }},
                                                        {8,
                                                         {
                                                             {-6170, {-1, 0, 0, 1}, 56936},
                                                             {239, {1, 0, -1, 0}, 4380},
                                                             {37, {1, 0, -1, 0}, 3559},
                                                             {-772, {-1, 0, 3, 2}, 16384},
                                                         }},
                                                        {9,
                                                         {
                                                             {-6133, {2, 0, -3, 1}, 18979},
                                                             {-1583769, {-1, 0, 0, 1}, 2920},
                                                             {-6133, {-1, 0, 0, 1}, 2372},
                                                             {-771, {1, 0, 3, 2}, 10923},
                                                         }},
                                                        {12,
                                                         {
                                                             {-6170, {-1, 0, 0, 1}, 56936},
                                                             {-1584008, {0, 0, -1, 1}, 8759},
                                                             {-6170, {0, 0, -1, 1}, 7117},
                                                             {775, {1, 0, 3, -2}, 32768},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, 1, 2, 1}, {1, 1, 1, 1}},
                                                    {
                                                        {8,
                                                         {
                                                             {37239, {-5, -2, 1, 6}, 9489},
                                                             {3168495, {1, 0, -1, 0}, 4380},
                                                             {12413, {1, 0, -1, 0}, 3559},
                                                             {-2318, {-2, 1, 4, 3}, 10923},
                                                         }},
                                                        {9,
                                                         {
                                                             {24789, {1, -1, -1, 1}, 28468},
                                                             {6337709, {-3, 1, -1, 3}, 876},
                                                             {24936, {-3, 1, -1, 3}, 712},
                                                             {-772, {1, 1, 1, 1}, 16384},
                                                         }},
                                                        {11,
                                                         {
                                                             {223026, {6, 5, -14, 3}, 4067},
                                                             {6338188, {-4, -1, 0, 5}, 626},
                                                             {25010, {-4, -1, 0, 5}, 508},
                                                             {-3087, {1, 2, 7, 4}, 4681},
                                                         }},
                                                        {0,
                                                         {
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                         }},
                                                        {0,
                                                         {
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                         }},
                                                        {0,
                                                         {
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                             {0, {0, 0, 0, 0}, 0},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, -1, -2, -1}, {1, 0, 1, 1}},
                                                    {
                                                        {1,
                                                         {
                                                             {-6170, {1, 0, 0, -1}, 56936},
                                                             {1584248, {0, 0, 1, -1}, 8759},
                                                             {6206, {0, 0, 1, -1}, 7117},
                                                             {1548, {1, 0, 3, -2}, 32768},
                                                         }},
                                                        {4,
                                                         {
                                                             {-43261, {-2, 0, 3, -1}, 18979},
                                                             {-1583769, {1, 0, 0, -1}, 2920},
                                                             {-6133, {1, 0, 0, -1}, 2372},
                                                             {-1544, {1, 0, 3, 2}, 10923},
                                                         }},
                                                        {5,
                                                         {
                                                             {-6170, {1, 0, 0, -1}, 56936},
                                                             {-3168017, {-1, 0, 1, 0}, 4380},
                                                             {-12339, {-1, 0, 1, 0}, 3559},
                                                             {-1545, {-1, 0, 3, 2}, 16384},
                                                         }},
                                                        {8,
                                                         {
                                                             {-6170, {-1, 0, 0, 1}, 56936},
                                                             {-3168017, {1, 0, -1, 0}, 4380},
                                                             {-12339, {1, 0, -1, 0}, 3559},
                                                             {1549, {-1, 0, 3, 2}, 16384},
                                                         }},
                                                        {9,
                                                         {
                                                             {-43261, {2, 0, -3, 1}, 18979},
                                                             {-1583769, {-1, 0, 0, 1}, 2920},
                                                             {-6133, {-1, 0, 0, 1}, 2372},
                                                             {1550, {1, 0, 3, 2}, 10923},
                                                         }},
                                                        {12,
                                                         {
                                                             {-6170, {-1, 0, 0, 1}, 56936},
                                                             {1584248, {0, 0, -1, 1}, 8759},
                                                             {6206, {0, 0, -1, 1}, 7117},
                                                             {-1546, {1, 0, 3, -2}, 32768},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, 1, 0, -1}, {1, 1, 0, 1}},
                                                    {
                                                        {1,
                                                         {
                                                             {-6170, {1, 0, 0, -1}, 56936},
                                                             {-3168017, {0, 1, 0, -1}, 4380},
                                                             {-12339, {0, 1, 0, -1}, 3559},
                                                             {-1545, {2, 3, 0, -1}, 16384},
                                                         }},
                                                        {2,
                                                         {
                                                             {6243, {-1, 3, 0, -2}, 18979},
                                                             {-1583769, {1, 0, 0, -1}, 2920},
                                                             {-6133, {1, 0, 0, -1}, 2372},
                                                             {1550, {2, 3, 0, 1}, 10923},
                                                         }},
                                                        {3,
                                                         {
                                                             {-6170, {1, 0, 0, -1}, 56936},
                                                             {1584248, {-1, 1, 0, 0}, 8759},
                                                             {6206, {-1, 1, 0, 0}, 7117},
                                                             {1548, {-2, 3, 0, 1}, 32768},
                                                         }},
                                                        {8,
                                                         {
                                                             {-6170, {-1, 0, 0, 1}, 56936},
                                                             {1584248, {1, -1, 0, 0}, 8759},
                                                             {6206, {1, -1, 0, 0}, 7117},
                                                             {-1546, {-2, 3, 0, 1}, 32768},
                                                         }},
                                                        {9,
                                                         {
                                                             {6243, {1, -3, 0, 2}, 18979},
                                                             {-1583769, {-1, 0, 0, 1}, 2920},
                                                             {-6133, {-1, 0, 0, 1}, 2372},
                                                             {-1544, {2, 3, 0, 1}, 10923},
                                                         }},
                                                        {10,
                                                         {
                                                             {-6170, {-1, 0, 0, 1}, 56936},
                                                             {-3168017, {0, -1, 0, 1}, 4380},
                                                             {-12339, {0, -1, 0, 1}, 3559},
                                                             {1549, {2, 3, 0, -1}, 16384},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, -1, -2, -3}, {1, 1, 1, 1}},
                                                    {
                                                        {8,
                                                         {
                                                             {-111274, {-5, -2, 1, 6}, 9489},
                                                             {-3168017, {1, 0, -1, 0}, 4380},
                                                             {-12339, {1, 0, -1, 0}, 3559},
                                                             {3, {-2, 1, 4, 3}, 10923},
                                                         }},
                                                        {12,
                                                         {
                                                             {-37091, {-1, -1, 1, 1}, 28468},
                                                             {-3168017, {1, -1, -1, 1}, 4380},
                                                             {-12339, {1, -1, -1, 1}, 3559},
                                                             {2, {-1, 3, 3, -1}, 16384},
                                                         }},
                                                        {14,
                                                         {
                                                             {-111274, {-6, -1, 2, 5}, 9489},
                                                             {-3168017, {0, -1, 0, 1}, 4380},
                                                             {-12339, {0, -1, 0, 1}, 3559},
                                                             {3, {3, 4, 1, -2}, 10923},
                                                         }},
                                                        {1,
                                                         {
                                                             {-111274, {6, 1, -2, -5}, 9489},
                                                             {-3168017, {0, 1, 0, -1}, 4380},
                                                             {-12339, {0, 1, 0, -1}, 3559},
                                                             {3, {3, 4, 1, -2}, 10923},
                                                         }},
                                                        {3,
                                                         {
                                                             {-37091, {1, 1, -1, -1}, 28468},
                                                             {-3168017, {-1, 1, 1, -1}, 4380},
                                                             {-12339, {-1, 1, 1, -1}, 3559},
                                                             {2, {-1, 3, 3, -1}, 16384},
                                                         }},
                                                        {7,
                                                         {
                                                             {-111274, {5, 2, -1, -6}, 9489},
                                                             {-3168017, {-1, 0, 1, 0}, 4380},
                                                             {-12339, {-1, 0, 1, 0}, 3559},
                                                             {3, {-2, 1, 4, 3}, 10923},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, 1, 0, 1}, {1, 1, 0, 1}},
                                                    {
                                                        {1,
                                                         {
                                                             {6206, {1, 0, 0, -1}, 56936},
                                                             {239, {0, 1, 0, -1}, 4380},
                                                             {37, {0, 1, 0, -1}, 3559},
                                                             {-772, {2, 3, 0, -1}, 16384},
                                                         }},
                                                        {2,
                                                         {
                                                             {30995, {-1, 3, 0, -2}, 18979},
                                                             {1584487, {1, 0, 0, -1}, 2920},
                                                             {6243, {1, 0, 0, -1}, 2372},
                                                             {777, {2, 3, 0, 1}, 10923},
                                                         }},
                                                        {3,
                                                         {
                                                             {6206, {1, 0, 0, -1}, 56936},
                                                             {1584248, {-1, 1, 0, 0}, 8759},
                                                             {6206, {-1, 1, 0, 0}, 7117},
                                                             {775, {-2, 3, 0, 1}, 32768},
                                                         }},
                                                        {8,
                                                         {
                                                             {6206, {-1, 0, 0, 1}, 56936},
                                                             {1584248, {1, -1, 0, 0}, 8759},
                                                             {6206, {1, -1, 0, 0}, 7117},
                                                             {-773, {-2, 3, 0, 1}, 32768},
                                                         }},
                                                        {9,
                                                         {
                                                             {30995, {1, -3, 0, 2}, 18979},
                                                             {1584487, {-1, 0, 0, 1}, 2920},
                                                             {6243, {-1, 0, 0, 1}, 2372},
                                                             {-771, {2, 3, 0, 1}, 10923},
                                                         }},
                                                        {10,
                                                         {
                                                             {6206, {-1, 0, 0, 1}, 56936},
                                                             {239, {0, -1, 0, 1}, 4380},
                                                             {37, {0, -1, 0, 1}, 3559},
                                                             {776, {2, 3, 0, -1}, 16384},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, 1, 2, 3}, {1, 0, 1, 1}},
                                                    {
                                                        {1,
                                                         {
                                                             {18582, {1, 0, 0, -1}, 56936},
                                                             {1584248, {0, 0, 1, -1}, 8759},
                                                             {6206, {0, 0, 1, -1}, 7117},
                                                             {1, {1, 0, 3, -2}, 32768},
                                                         }},
                                                        {4,
                                                         {
                                                             {55747, {-2, 0, 3, -1}, 18979},
                                                             {4752743, {1, 0, 0, -1}, 2920},
                                                             {18619, {1, 0, 0, -1}, 2372},
                                                             {3, {1, 0, 3, 2}, 10923},
                                                         }},
                                                        {5,
                                                         {
                                                             {18582, {1, 0, 0, -1}, 56936},
                                                             {3168495, {-1, 0, 1, 0}, 4380},
                                                             {12413, {-1, 0, 1, 0}, 3559},
                                                             {2, {-1, 0, 3, 2}, 16384},
                                                         }},
                                                        {8,
                                                         {
                                                             {18582, {-1, 0, 0, 1}, 56936},
                                                             {3168495, {1, 0, -1, 0}, 4380},
                                                             {12413, {1, 0, -1, 0}, 3559},
                                                             {2, {-1, 0, 3, 2}, 16384},
                                                         }},
                                                        {9,
                                                         {
                                                             {55747, {2, 0, -3, 1}, 18979},
                                                             {4752743, {-1, 0, 0, 1}, 2920},
                                                             {18619, {-1, 0, 0, 1}, 2372},
                                                             {3, {1, 0, 3, 2}, 10923},
                                                         }},
                                                        {12,
                                                         {
                                                             {18582, {-1, 0, 0, 1}, 56936},
                                                             {1584248, {0, 0, -1, 1}, 8759},
                                                             {6206, {0, 0, -1, 1}, 7117},
                                                             {1, {1, 0, 3, -2}, 32768},
                                                         }},
                                                    }});
  LAYOUT_VALID_TO_LATCOMB_CONSTS_ENCODER.push_back({{{0, -1, 0, 1}, {1, 1, 0, 1}},
                                                    {
                                                        {1,
                                                         {
                                                             {6206, {1, 0, 0, -1}, 56936},
                                                             {3168495, {0, 1, 0, -1}, 4380},
                                                             {12413, {0, 1, 0, -1}, 3559},
                                                             {1549, {2, 3, 0, -1}, 16384},
                                                         }},
                                                        {2,
                                                         {
                                                             {-6133, {-1, 3, 0, -2}, 18979},
                                                             {1584487, {1, 0, 0, -1}, 2920},
                                                             {6243, {1, 0, 0, -1}, 2372},
                                                             {-1544, {2, 3, 0, 1}, 10923},
                                                         }},
                                                        {3,
                                                         {
                                                             {6206, {1, 0, 0, -1}, 56936},
                                                             {-1584008, {-1, 1, 0, 0}, 8759},
                                                             {-6170, {-1, 1, 0, 0}, 7117},
                                                             {-1546, {-2, 3, 0, 1}, 32768},
                                                         }},
                                                        {8,
                                                         {
                                                             {6206, {-1, 0, 0, 1}, 56936},
                                                             {-1584008, {1, -1, 0, 0}, 8759},
                                                             {-6170, {1, -1, 0, 0}, 7117},
                                                             {1548, {-2, 3, 0, 1}, 32768},
                                                         }},
                                                        {9,
                                                         {
                                                             {-6133, {1, -3, 0, 2}, 18979},
                                                             {1584487, {-1, 0, 0, 1}, 2920},
                                                             {6243, {-1, 0, 0, 1}, 2372},
                                                             {1550, {2, 3, 0, 1}, 10923},
                                                         }},
                                                        {10,
                                                         {
                                                             {6206, {-1, 0, 0, 1}, 56936},
                                                             {3168495, {0, -1, 0, 1}, 4380},
                                                             {12413, {0, -1, 0, 1}, 3559},
                                                             {-1545, {2, 3, 0, -1}, 16384},
                                                         }},
                                                    }});
}
