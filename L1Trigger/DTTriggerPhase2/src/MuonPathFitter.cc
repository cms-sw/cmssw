#include "L1Trigger/DTTriggerPhase2/interface/MuonPathFitter.h"
#include <cmath>
#include <memory>

using namespace edm;
using namespace std;
using namespace cmsdt;

// ============================================================================
// Constructors and destructor
// ============================================================================
MuonPathFitter::MuonPathFitter(const ParameterSet &pset,
                               edm::ConsumesCollector &iC,
                               std::shared_ptr<GlobalCoordsObtainer> &globalcoordsobtainer)
    : MuonPathAnalyzer(pset, iC), debug_(pset.getUntrackedParameter<bool>("debug")) {
  if (debug_)
    LogDebug("MuonPathFitter") << "MuonPathAnalyzer: constructor";

  //shift phi
  int rawId;
  shift_filename_ = pset.getParameter<edm::FileInPath>("shift_filename");
  std::ifstream ifin3(shift_filename_.fullPath());
  double shift;
  if (ifin3.fail()) {
    throw cms::Exception("Missing Input File")
        << "MuonPathFitter::MuonPathFitter() -  Cannot find " << shift_filename_.fullPath();
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

  dtGeomH = iC.esConsumes<DTGeometry, MuonGeometryRecord, edm::Transition::BeginRun>();
  globalcoordsobtainer_ = globalcoordsobtainer;
}

MuonPathFitter::~MuonPathFitter() {
  if (debug_)
    LogDebug("MuonPathFitter") << "MuonPathAnalyzer: destructor";
}

// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================

//------------------------------------------------------------------
//--- Metodos privados
//------------------------------------------------------------------

fit_common_out_t MuonPathFitter::fit(fit_common_in_t fit_common_in,
                                     int XI_WIDTH,
                                     int COEFF_WIDTH_T0,
                                     int COEFF_WIDTH_POSITION,
                                     int COEFF_WIDTH_SLOPE,
                                     int PRECISSION_T0,
                                     int PRECISSION_POSITION,
                                     int PRECISSION_SLOPE,
                                     int PROD_RESIZE_T0,
                                     int PROD_RESIZE_POSITION,
                                     int PROD_RESIZE_SLOPE,
                                     int MAX_DRIFT_TDC,
                                     int sl) {
  const int PARTIALS_PRECISSION = 4;
  const int PARTIALS_SHR_T0 = PRECISSION_T0 - PARTIALS_PRECISSION;
  const int PARTIALS_SHR_POSITION = PRECISSION_POSITION - PARTIALS_PRECISSION;
  const int PARTIALS_SHR_SLOPE = PRECISSION_SLOPE - PARTIALS_PRECISSION;
  const int PARTIALS_WIDTH_T0 = PROD_RESIZE_T0 - PARTIALS_SHR_T0;
  const int PARTIALS_WIDTH_POSITION = PROD_RESIZE_POSITION - PARTIALS_SHR_POSITION;
  const int PARTIALS_WIDTH_SLOPE = PROD_RESIZE_SLOPE - PARTIALS_SHR_SLOPE;

  const int WIDTH_TO_PREC = 11 + PARTIALS_PRECISSION;
  const int WIDTH_SLOPE_PREC = 14 + PARTIALS_PRECISSION;
  const int WIDTH_POSITION_PREC = WIDTH_SLOPE_PREC + 1;

  const int SEMICHAMBER_H_PRECISSION = 13 + PARTIALS_PRECISSION;
  const float SEMICHAMBER_H_REAL = ((235. / 2.) / (16. * 6.5)) * std::pow(2, SEMICHAMBER_H_PRECISSION);
  const int SEMICHAMBER_H = (int)SEMICHAMBER_H_REAL;  // signed(SEMICHAMBER_H_WIDTH-1 downto 0)

  const int SEMICHAMBER_RES_SHR = SEMICHAMBER_H_PRECISSION;

  const int LYRANDAHALF_RES_SHR = 4;

  const int CHI2_CALC_RES_BITS = 7;

  /*******************************
            clock cycle 1
  *******************************/
  std::vector<int> normalized_times;
  std::vector<int> normalized_wirepos;

  for (int i = 0; i < 2 * NUM_LAYERS; i++) {
    // normalized times
    // this should be resized to an unsigned of 10 bits (max drift time ~508 TDC counts, using 9+1 to include tolerance)
    // leaving it as an integer for now
    // we are obtaining the difference as the difference in BX + the LS bits from the hit time

    if (fit_common_in.hits_valid[i] == 1) {
      int dif_bx = (fit_common_in.hits[i].ti >> (WIDTH_FULL_TIME - WIDTH_COARSED_TIME)) - fit_common_in.coarse_bctr;

      int tmp_norm_time = (dif_bx << (WIDTH_FULL_TIME - WIDTH_COARSED_TIME)) +
                          (fit_common_in.hits[i].ti % (int)std::pow(2, WIDTH_FULL_TIME - WIDTH_COARSED_TIME));
      // resize test
      // this has implications in the FW (reducing number of bits).
      // we keep here the int as it is, but we do the same check done in the fw
      std::vector<int> tmp_dif_bx_vector;
      vhdl_int_to_unsigned(dif_bx, tmp_dif_bx_vector);
      vhdl_resize_unsigned(tmp_dif_bx_vector, 12);
      if (!vhdl_resize_unsigned_ok(tmp_dif_bx_vector, WIDTH_DIFBX))
        return fit_common_out_t();

      normalized_times.push_back(tmp_norm_time);
      int tmp_wirepos = fit_common_in.hits[i].wp - (fit_common_in.coarse_wirepos << WIREPOS_NORM_LSB_IGNORED);
      // resize test
      std::vector<int> tmp_wirepos_vector;
      vhdl_int_to_signed(tmp_wirepos, tmp_wirepos_vector);
      vhdl_resize_signed(tmp_wirepos_vector, WIREPOS_WIDTH);
      if (!vhdl_resize_signed_ok(tmp_wirepos_vector, XI_WIDTH))
        return fit_common_out_t();

      normalized_wirepos.push_back(tmp_wirepos);
    } else {  // dummy hit
      normalized_times.push_back(-1);
      normalized_wirepos.push_back(-1);
    }
  }

  /*******************************
            clock cycle 2
  *******************************/

  std::vector<int> xi_arr;
  // min and max times are computed throught several clk cycles in the fw,
  // here we compute it at once
  int min_hit_time = 999999, max_hit_time = 0;
  for (int i = 0; i < 2 * NUM_LAYERS; i++) {
    if (fit_common_in.hits_valid[i] == 1) {
      // calculate xi array
      auto tmp_xi_incr = normalized_wirepos[i];
      tmp_xi_incr += (-1 + 2 * fit_common_in.lateralities[i]) * normalized_times[i];

      // resize test
      std::vector<int> tmp_xi_incr_vector;
      vhdl_int_to_signed(tmp_xi_incr, tmp_xi_incr_vector);
      vhdl_resize_signed(tmp_xi_incr_vector, XI_WIDTH + 1);
      if (!vhdl_resize_signed_ok(tmp_xi_incr_vector, XI_WIDTH))
        return fit_common_out_t();
      xi_arr.push_back(tmp_xi_incr);

      // calculate min and max times
      if (normalized_times[i] < min_hit_time) {
        min_hit_time = normalized_times[i];
      }
      if (normalized_times[i] > max_hit_time) {
        max_hit_time = normalized_times[i];
      }
    } else {
      xi_arr.push_back(-1);
    }
  }

  /*******************************
            clock cycle 3
  *******************************/

  std::vector<int> products_t0;
  std::vector<int> products_position;
  std::vector<int> products_slope;
  for (int i = 0; i < 2 * NUM_LAYERS; i++) {
    if (fit_common_in.hits_valid[i] == 0) {
      products_t0.push_back(-1);
      products_position.push_back(-1);
      products_slope.push_back(-1);
    } else {
      products_t0.push_back(xi_arr[i] * vhdl_signed_to_int(fit_common_in.coeffs.t0[i]));
      products_position.push_back(xi_arr[i] * vhdl_signed_to_int(fit_common_in.coeffs.position[i]));
      products_slope.push_back(xi_arr[i] * vhdl_signed_to_int(fit_common_in.coeffs.slope[i]));
    }
  }

  /*******************************
            clock cycle 4
  *******************************/
  // Do the 8 element sums
  int t0_prec = 0, position_prec = 0, slope_prec = 0;
  for (int i = 0; i < 2 * NUM_LAYERS; i++) {
    if (fit_common_in.hits_valid[i] == 0) {
      continue;
    } else {
      t0_prec += products_t0[i] >> PARTIALS_SHR_T0;
      position_prec += products_position[i] >> PARTIALS_SHR_POSITION;
      slope_prec += products_slope[i] >> PARTIALS_SHR_SLOPE;
    }
  }

  /*******************************
            clock cycle 5
  *******************************/
  // Do resize tests for the computed sums with full precision
  std::vector<int> t0_prec_vector, position_prec_vector, slope_prec_vector;
  vhdl_int_to_signed(t0_prec, t0_prec_vector);

  vhdl_resize_signed(t0_prec_vector, PARTIALS_WIDTH_T0);
  if (!vhdl_resize_signed_ok(t0_prec_vector, WIDTH_TO_PREC))
    return fit_common_out_t();

  vhdl_int_to_signed(position_prec, position_prec_vector);
  vhdl_resize_signed(position_prec_vector, PARTIALS_WIDTH_POSITION);
  if (!vhdl_resize_signed_ok(position_prec_vector, WIDTH_POSITION_PREC))
    return fit_common_out_t();

  vhdl_int_to_signed(slope_prec, slope_prec_vector);
  vhdl_resize_signed(slope_prec_vector, PARTIALS_WIDTH_SLOPE);
  if (!vhdl_resize_signed_ok(slope_prec_vector, WIDTH_SLOPE_PREC))
    return fit_common_out_t();

  /*******************************
            clock cycle 6
  *******************************/
  // Round the fitting parameters to the final resolution;
  // in vhdl something more sofisticated is done, here we do a float division, round
  // and cast again to integer

  int norm_t0 = ((t0_prec >> (PARTIALS_PRECISSION - 1)) + 1) >> 1;
  int norm_position = ((position_prec >> (PARTIALS_PRECISSION - 1)) + 1) >> 1;
  int norm_slope = ((slope_prec >> (PARTIALS_PRECISSION - 1)) + 1) >> 1;

  // Calculate the (-xi) + pos (+/-) t0, which only is lacking the slope term to become the residuals
  std::vector<int> res_partials_arr;
  for (int i = 0; i < 2 * NUM_LAYERS; i++) {
    if (fit_common_in.hits_valid[i] == 0) {
      res_partials_arr.push_back(-1);
    } else {
      int tmp_position_prec = position_prec - (xi_arr[i] << PARTIALS_PRECISSION);
      // rounding
      tmp_position_prec += std::pow(2, PARTIALS_PRECISSION - 1);

      tmp_position_prec += (-1 + 2 * fit_common_in.lateralities[i]) * t0_prec;
      res_partials_arr.push_back(tmp_position_prec);
    }
  }

  // calculate the { slope x semichamber, slope x 1.5 layers, slope x 0.5 layers }
  // these 3 values are later combined with different signs to get the slope part
  // of the residual for each of the layers.
  int slope_x_halfchamb = (((long int)slope_prec * (long int)SEMICHAMBER_H)) >> SEMICHAMBER_RES_SHR;
  if (sl == 2)
    slope_x_halfchamb = 0;
  int slope_x_3semicells = (slope_prec * 3) >> LYRANDAHALF_RES_SHR;
  int slope_x_1semicell = (slope_prec * 1) >> LYRANDAHALF_RES_SHR;

  /*******************************
            clock cycle 7
  *******************************/
  // Complete the residuals calculation by constructing the slope term (1/2)
  for (int i = 0; i < 2 * NUM_LAYERS; i++) {
    if (fit_common_in.hits_valid[i] == 1) {
      if (i % 4 == 0)
        res_partials_arr[i] -= slope_x_3semicells;
      else if (i % 4 == 1)
        res_partials_arr[i] -= slope_x_1semicell;
      else if (i % 4 == 2)
        res_partials_arr[i] += slope_x_1semicell;
      else
        res_partials_arr[i] += slope_x_3semicells;
    }
  }

  /*******************************
            clock cycle 8
  *******************************/
  // Complete the residuals calculation by constructing the slope term (2/2)
  std::vector<int> residuals, position_prec_arr;
  for (int i = 0; i < 2 * NUM_LAYERS; i++) {
    if (fit_common_in.hits_valid[i] == 0) {
      residuals.push_back(-1);
      position_prec_arr.push_back(-1);
    } else {
      int tmp_position_prec = res_partials_arr[i];
      tmp_position_prec += (-1 + 2 * (int)(i >= NUM_LAYERS)) * slope_x_halfchamb;
      position_prec_arr.push_back(tmp_position_prec);
      residuals.push_back(abs(tmp_position_prec >> PARTIALS_PRECISSION));
    }
  }

  // minimum and maximum fit t0
  int min_t0 = max_hit_time - MAX_DRIFT_TDC - T0_CUT_TOLERANCE;
  int max_t0 = min_hit_time + T0_CUT_TOLERANCE;

  /*******************************
            clock cycle 9
  *******************************/
  // Prepare addition of coarse_offset to T0 (T0 de-normalization)
  int t0_fine = norm_t0 & (int)(std::pow(2, 5) - 1);
  int t0_bx_sign = ((int)(norm_t0 < 0)) * 1;
  int t0_bx_abs = abs(norm_t0 >> 5);

  // De-normalize Position and slope
  int position = (fit_common_in.coarse_wirepos << WIREPOS_NORM_LSB_IGNORED) + norm_position;
  int slope = norm_slope;

  // Apply T0 cuts
  if (norm_t0 < min_t0)
    return fit_common_out_t();
  if (norm_t0 > max_t0)
    return fit_common_out_t();

  // square the residuals
  std::vector<int> squared_residuals;
  for (int i = 0; i < 2 * NUM_LAYERS; i++) {
    if (fit_common_in.hits_valid[i] == 0) {
      squared_residuals.push_back(-1);
    } else {
      squared_residuals.push_back(residuals[i] * residuals[i]);
    }
  }

  // check for residuals overflow
  for (int i = 0; i < 2 * NUM_LAYERS; i++) {
    if (fit_common_in.hits_valid[i] == 1) {
      std::vector<int> tmp_vector;
      int tmp_position_prec = (position_prec_arr[i] >> PARTIALS_PRECISSION);
      vhdl_int_to_signed(tmp_position_prec, tmp_vector);
      vhdl_resize_signed(tmp_vector, WIDTH_POSITION_PREC);
      if (!vhdl_resize_signed_ok(tmp_vector, CHI2_CALC_RES_BITS + 1))
        return fit_common_out_t();
      // Commented for now, maybe later we need to do something here
      // if ((tmp_position_prec / (int) std::pow(2, CHI2_CALC_RES_BITS)) > 0)
      // return fit_common_out_t();
    }
  }

  /*******************************
        clock cycle 10, 11, 12
  *******************************/
  int t0 = t0_fine;
  t0 += (fit_common_in.coarse_bctr - (-1 + 2 * t0_bx_sign) * t0_bx_abs) * (int)std::pow(2, 5);

  int chi2 = 0;
  for (int i = 0; i < 2 * NUM_LAYERS; i++) {
    if (fit_common_in.hits_valid[i] == 1) {
      chi2 += squared_residuals[i];
    }
  }

  // Impose the thresholds

  if (chi2 / 16 >= (int)round(chi2Th_ * (std::pow((float)MAX_DRIFT_TDC / ((float)CELL_SEMILENGTH / 10.), 2)) / 16.))
    return fit_common_out_t();

  fit_common_out_t fit_common_out;
  fit_common_out.position = position;
  fit_common_out.slope = slope;
  fit_common_out.t0 = t0;
  fit_common_out.chi2 = chi2;
  fit_common_out.valid_fit = 1;

  return fit_common_out;
}

coeffs_t MuonPathFitter::RomDataConvert(std::vector<int> slv,
                                        short COEFF_WIDTH_T0,
                                        short COEFF_WIDTH_POSITION,
                                        short COEFF_WIDTH_SLOPE,
                                        short LOLY,
                                        short HILY) {
  coeffs_t res;
  int ctr = 0;
  for (int i = LOLY; i <= HILY; i++) {
    res.t0[i] = vhdl_slice(slv, COEFF_WIDTH_T0 + ctr - 1, ctr);
    vhdl_resize_unsigned(res.t0[i], GENERIC_COEFF_WIDTH);
    res.t0[i] = vhdl_slice(res.t0[i], COEFF_WIDTH_T0 - 1, 0);
    ctr += COEFF_WIDTH_T0;
  }
  for (int i = LOLY; i <= HILY; i++) {
    res.position[i] = vhdl_slice(slv, COEFF_WIDTH_POSITION + ctr - 1, ctr);
    vhdl_resize_unsigned(res.position[i], GENERIC_COEFF_WIDTH);
    res.position[i] = vhdl_slice(res.position[i], COEFF_WIDTH_POSITION - 1, 0);
    ctr += COEFF_WIDTH_POSITION;
  }
  for (int i = LOLY; i <= HILY; i++) {
    res.slope[i] = vhdl_slice(slv, COEFF_WIDTH_SLOPE + ctr - 1, ctr);
    vhdl_resize_unsigned(res.slope[i], GENERIC_COEFF_WIDTH);
    res.slope[i] = vhdl_slice(res.slope[i], COEFF_WIDTH_SLOPE - 1, 0);
    ctr += COEFF_WIDTH_SLOPE;
  }
  return res;
}
