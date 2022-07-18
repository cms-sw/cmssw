#ifndef L1TMuonEndCap_TrackTools_h
#define L1TMuonEndCap_TrackTools_h

#include <cmath>

namespace emtf {

  // Please refers to DN-2015/017 for uGMT conventions

  int calc_ring(int station, int csc_ID, int strip);

  int calc_chamber(int station, int sector, int subsector, int ring, int csc_ID);

  int calc_uGMT_chamber(int csc_ID, int subsector, int neighbor, int station);

  // CSC trigger sector & CSC ID

  int get_trigger_sector(int ring, int station, int chamber);

  int get_trigger_csc_ID(int ring, int station, int chamber);

  // CSC max strip & max wire

  std::pair<int, int> get_csc_max_strip_and_wire(int station, int ring);

  // CSC max pattern & max quality

  std::pair<int, int> get_csc_max_pattern_and_quality(int station, int ring);

  // CSC max slope

  int get_csc_max_slope(int station, int ring, bool useRun3CCLUT_OTMB, bool useRun3CCLUT_TMB);

  // ___________________________________________________________________________
  // coordinate ranges: phi[-180, 180] or [-pi, pi], theta[0, 90] or [0, pi/2]
  inline double wrap_phi_deg(double deg) {
    while (deg < -180.)
      deg += 360.;
    while (deg >= +180.)
      deg -= 360.;
    return deg;
  }

  inline double wrap_phi_rad(double rad) {
    while (rad < -M_PI)
      rad += 2. * M_PI;
    while (rad >= +M_PI)
      rad -= 2. * M_PI;
    return rad;
  }

  inline double wrap_theta_deg(double deg) {
    deg = std::abs(deg);
    while (deg >= 180.)
      deg -= 180.;
    if (deg >= 180. / 2.)
      deg = 180. - deg;
    return deg;
  }

  inline double wrap_theta_rad(double rad) {
    rad = std::abs(rad);
    while (rad >= M_PI)
      rad -= M_PI;
    if (rad >= M_PI / 2.)
      rad = M_PI - rad;
    return rad;
  }

  // ___________________________________________________________________________
  // radians, degrees
  inline double deg_to_rad(double deg) {
    constexpr double factor = M_PI / 180.;
    return deg * factor;
  }

  inline double rad_to_deg(double rad) {
    constexpr double factor = 180. / M_PI;
    return rad * factor;
  }

  // ___________________________________________________________________________
  // pt
  inline double calc_pt(int bits) {
    double pt = static_cast<double>(bits);
    pt = 0.5 * (pt - 1);
    return pt;
  }

  inline int calc_pt_GMT(double val) {
    val = (val * 2) + 1;
    int gmt_pt = static_cast<int>(std::round(val));
    gmt_pt = (gmt_pt > 511) ? 511 : gmt_pt;
    return gmt_pt;
  }

  // ___________________________________________________________________________
  // eta
  inline double calc_eta(int bits) {
    double eta = static_cast<double>(bits);
    eta *= 0.010875;
    return eta;
  }

  //inline double calc_eta_corr(int bits, int endcap) {  // endcap [-1,+1]
  //  bits = (endcap == -1) ? bits+1 : bits;
  //  double eta = static_cast<double>(bits);
  //  eta *= 0.010875;
  //  return eta;
  //}

  inline double calc_eta_from_theta_rad(double theta_rad) {
    double eta = -1. * std::log(std::tan(theta_rad / 2.));
    return eta;
  }

  inline double calc_eta_from_theta_deg(double theta_deg, int endcap) {  // endcap [-1,+1]
    double theta_rad = deg_to_rad(wrap_theta_deg(theta_deg));            // put theta in [0, 90] range
    double eta = calc_eta_from_theta_rad(theta_rad);
    eta = (endcap == -1) ? -eta : eta;
    return eta;
  }

  inline int calc_eta_GMT(double val) {
    val /= 0.010875;
    int gmt_eta = static_cast<int>(std::round(val));
    return gmt_eta;
  }

  // ___________________________________________________________________________
  // theta
  inline double calc_theta_deg_from_int(int theta_int) {
    double theta = static_cast<double>(theta_int);
    theta = theta * (45.0 - 8.5) / 128. + 8.5;
    return theta;
  }

  inline double calc_theta_rad_from_int(int theta_int) { return deg_to_rad(calc_theta_deg_from_int(theta_int)); }

  inline double calc_theta_rad(double eta) {
    double theta_rad = 2. * std::atan(std::exp(-1. * eta));
    return theta_rad;
  }

  inline double calc_theta_deg(double eta) { return rad_to_deg(calc_theta_rad(eta)); }

  inline int calc_theta_int(double theta, int endcap) {  // theta in deg, endcap [-1,+1]
    theta = (endcap == -1) ? (180. - theta) : theta;
    theta = (theta - 8.5) * 128. / (45.0 - 8.5);
    int theta_int = static_cast<int>(std::round(theta));
    return theta_int;
  }

  inline int calc_theta_int_rpc(double theta, int endcap) {  // theta in deg, endcap [-1,+1]
    theta = (endcap == -1) ? (180. - theta) : theta;
    theta = (theta - 8.5) * (128. / 4.) / (45.0 - 8.5);  // 4x coarser resolution
    int theta_int = static_cast<int>(std::round(theta));
    return theta_int;
  }

  inline double calc_theta_rad_from_eta(double eta) {
    double theta = std::atan2(1.0, std::sinh(eta));  // cot(theta) = sinh(eta)
    return theta;
  }

  inline double calc_theta_deg_from_eta(double eta) { return rad_to_deg(calc_theta_rad_from_eta(eta)); }

  // ___________________________________________________________________________
  // phi
  inline double calc_phi_glob_deg(double loc, int sector) {  // loc in deg, sector [1-6]
    double glob = loc + 15. + (60. * (sector - 1));
    glob = (glob < 180.) ? glob : glob - 360.;
    return glob;
  }

  inline double calc_phi_glob_rad(double loc, int sector) {  // loc in rad, sector [1-6]
    return deg_to_rad(calc_phi_glob_deg(rad_to_deg(loc), sector));
  }

  inline double calc_phi_loc_deg(int bits) {
    double loc = static_cast<double>(bits);
    loc = (loc / 60.) - 22.;
    return loc;
  }

  inline double calc_phi_loc_rad(int bits) { return deg_to_rad(calc_phi_loc_deg(bits)); }

  //inline double calc_phi_loc_deg_corr(int bits, int endcap) {  // endcap [-1,+1]
  //  double loc = static_cast<double>(bits);
  //  loc = (loc/60.) - 22.;
  //  loc = (endcap == -1) ? loc - (36./60.) : loc - (28./60.);
  //  return loc;
  //}

  //inline double calc_phi_loc_rad_corr(int bits, int endcap) {  // endcap [-1,+1]
  //  return deg_to_rad(calc_phi_loc_deg_corr(bits, endcap));
  //}

  inline double calc_phi_loc_deg_from_glob(double glob, int sector) {  // glob in deg, sector [1-6]
    glob = wrap_phi_deg(glob);                                         // put phi in [-180,180] range
    double loc = glob - 15. - (60. * (sector - 1));
    return loc;
  }

  inline int calc_phi_loc_int(double glob, int sector) {  // glob in deg, sector [1-6]
    double loc = calc_phi_loc_deg_from_glob(glob, sector);
    loc = ((loc + 22.) < 0.) ? loc + 360. : loc;
    loc = (loc + 22.) * 60.;
    int phi_int = static_cast<int>(std::round(loc));
    return phi_int;
  }

  inline int calc_phi_loc_int_rpc(double glob, int sector) {  // glob in deg, sector [1-6]
    double loc = calc_phi_loc_deg_from_glob(glob, sector);
    loc = ((loc + 22.) < 0.) ? loc + 360. : loc;
    loc = (loc + 22.) * 60. / 4.;  // 4x coarser resolution
    int phi_int = static_cast<int>(std::round(loc));
    return phi_int;
  }

  inline double calc_phi_GMT_deg(int bits) {
    double phi = static_cast<double>(bits);
    phi = (phi * 360. / 576.) + (180. / 576.);
    return phi;
  }

  //inline double calc_phi_GMT_deg_corr(int bits) {  // AWB mod 09.02.16
  //  return (bits * 0.625 * 1.0208) + 0.3125 * 1.0208 + 0.552;
  //}

  inline double calc_phi_GMT_rad(int bits) { return deg_to_rad(calc_phi_GMT_deg(bits)); }

  inline int calc_phi_GMT_int(double val) {  // phi in deg
    val = wrap_phi_deg(val);                 // put phi in [-180,180] range
    val = (val - 180. / 576.) / (360. / 576.);
    int gmt_phi = static_cast<int>(std::round(val));
    return gmt_phi;
  }

}  // namespace emtf

#endif
