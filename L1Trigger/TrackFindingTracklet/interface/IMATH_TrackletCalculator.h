#ifndef L1Trigger_TrackFindingTracklet_interface_IMATH_TrackletCalculator_h
#define L1Trigger_TrackFindingTracklet_interface_IMATH_TrackletCalculator_h

#include "Settings.h"
#include "imath.h"

//
// Constants used:
//   dphisector
//   rmaxL6
//   zmaxD5
//   rmaxdisk
//   kr, kphi1, kz
//
//   rmean[], zmean[]

namespace trklet {
  class Globals;

  class IMATH_TrackletCalculator {
  public:
    IMATH_TrackletCalculator(Settings const& settings, imathGlobals* globals, int i1, int i2)
        : settings_(settings), globals_(globals) {
      if (settings_.debugTracklet()) {
        edm::LogVerbatim("Tracklet") << "=============================================";
        char s[1024];
        snprintf(s, 1024, "IMATH Tracklet Calculator %i %i dphisector = %f", i1, i2, settings_.dphisector());
        edm::LogVerbatim("Tracklet") << s;
        snprintf(s, 1024, "rmaxL6 = %f, zmaxD5 = %f", settings_.rmax(5), settings_.zmax(4));
        edm::LogVerbatim("Tracklet") << s;
        snprintf(
            s, 1024, "      stub Ks: kr, kphi1, kz = %g, %g, %g", settings_.kr(), settings_.kphi1(), settings_.kz());
        edm::LogVerbatim("Tracklet") << s;
        snprintf(s,
                 1024,
                 "  tracklet Ks: krinvpars, kphi0pars, ktpars, kzpars = %g, %g, %g, %g",
                 settings_.kphi1() / settings_.kr() * pow(2, settings_.rinv_shift()),
                 settings_.kphi1() * pow(2, settings_.phi0_shift()),
                 settings_.kz() / settings_.kr() * pow(2, settings_.t_shift()),
                 settings_.kz() * pow(2, settings_.z0_shift()));
        edm::LogVerbatim("Tracklet") << s;
        snprintf(s,
                 1024,
                 "layer proj Ks: kphiproj456, kphider, kzproj, kzder = %g, %g, %g, %g",
                 settings_.kphi1() * pow(2, settings_.SS_phiL_shift()),
                 settings_.kphi1() / settings_.kr() * pow(2, settings_.SS_phiderL_shift()),
                 settings_.kz() * pow(2, settings_.PS_zL_shift()),
                 settings_.kz() / settings_.kr() * pow(2, settings_.PS_zderL_shift()));
        edm::LogVerbatim("Tracklet") << s;
        snprintf(s,
                 1024,
                 " disk proj Ks: kphiprojdisk, kphiprojderdisk, krprojdisk, krprojderdisk = %g, %g, %g, %g",
                 settings_.kphi1() * pow(2, settings_.SS_phiD_shift()),
                 settings_.kphi1() / settings_.kr() * pow(2, settings_.SS_phiderD_shift()),
                 settings_.kr() * pow(2, settings_.PS_rD_shift()),
                 settings_.kr() / settings_.kz() * pow(2, settings_.PS_rderD_shift()));
        edm::LogVerbatim("Tracklet") << s;
        edm::LogVerbatim("Tracklet") << "=============================================";

        snprintf(s, 1024, "initializing 1/dr LUT %f %f", settings_.rmean(i1 - 1), settings_.rmean(i2 - 1));
        edm::LogVerbatim("Tracklet") << s;
      }

      drinv.initLUT(settings_.rmean(i2 - 1) - settings_.rmean(i1 - 1));
      r1mean.set_fval(settings_.rmean(i1 - 1));
      r2mean.set_fval(settings_.rmean(i2 - 1));
      r12mean.set_fval(settings_.rmean(i1 - 1) + settings_.rmean(i2 - 1));

      if (i1 == 1)
        z0_final.add_cut(&z0_final_L1_cut);
      else
        z0_final.add_cut(&z0_final_cut);

      valid_phiL_0.add_cut(&t_layer_cut);
      valid_phiL_1.add_cut(&t_layer_cut);
      valid_phiL_2.add_cut(&t_layer_cut);
      valid_phiL_3.add_cut(&t_layer_cut);

      valid_der_phiL.add_cut(&t_layer_cut);

      valid_zL_0.add_cut(&t_layer_cut);
      valid_zL_1.add_cut(&t_layer_cut);
      valid_zL_2.add_cut(&t_layer_cut);
      valid_zL_3.add_cut(&t_layer_cut);

      valid_der_zL.add_cut(&t_layer_cut);

      valid_phiD_0.add_cut(&t_disk_cut_left);
      valid_phiD_1.add_cut(&t_disk_cut_left);
      valid_phiD_2.add_cut(&t_disk_cut_left);
      valid_phiD_3.add_cut(&t_disk_cut_left);
      valid_phiD_4.add_cut(&t_disk_cut_left);

      valid_der_phiD.add_cut(&t_disk_cut_left);

      valid_rD_0.add_cut(&t_disk_cut_left);
      valid_rD_1.add_cut(&t_disk_cut_left);
      valid_rD_2.add_cut(&t_disk_cut_left);
      valid_rD_3.add_cut(&t_disk_cut_left);
      valid_rD_4.add_cut(&t_disk_cut_left);

      valid_der_rD.add_cut(&t_disk_cut_left);

      valid_phiD_0.add_cut(&t_disk_cut_right);
      valid_phiD_1.add_cut(&t_disk_cut_right);
      valid_phiD_2.add_cut(&t_disk_cut_right);
      valid_phiD_3.add_cut(&t_disk_cut_right);
      valid_phiD_4.add_cut(&t_disk_cut_right);

      valid_der_phiD.add_cut(&t_disk_cut_right);

      valid_rD_0.add_cut(&t_disk_cut_right);
      valid_rD_1.add_cut(&t_disk_cut_right);
      valid_rD_2.add_cut(&t_disk_cut_right);
      valid_rD_3.add_cut(&t_disk_cut_right);
      valid_rD_4.add_cut(&t_disk_cut_right);

      valid_der_rD.add_cut(&t_disk_cut_right);
    }

    ~IMATH_TrackletCalculator() = default;

    Settings const& settings_;

    imathGlobals* globals_;

    //max values
    const double dz_max = 50.;
    const double delta0_max = 0.005;
    const double a2_max = 3.;
    const double a2a_max = 0.1;
    const double x6a_max = 0.02;
    const double x6m_max = 2.;
    const double x8_max = 1.;
    const double x13_max = 300.;
    const double x22_max = 0.3;
    const double x23_max = 200.;
    const double t_max = 4.;
    const double z0_max = 20.;
    const double der_phiD_max = 0.002;
    const double t_disk_min = 1;
    const double t_disk_max = 4;
    const double t_layer_max = 2.5;

    //constants
    VarParam plus1{globals_, "plus1", 1., 10};
    VarParam plus2{globals_, "plus2", 2., 10};
    VarParam minus1{globals_, "minus1", -1., 10};

    VarParam r1mean{globals_, "r1mean", "Kr", settings_.rmax(N_LAYER - 1), settings_.kr()};
    VarParam r2mean{globals_, "r2mean", "Kr", settings_.rmax(N_LAYER - 1), settings_.kr()};
    VarParam r12mean{globals_, "r12mean", "Kr", 2 * settings_.rmax(N_DISK - 1), settings_.kr()};

    //inputs
    VarDef r1{globals_, "r1", "Kr", settings_.drmax(), settings_.kr()};
    VarDef r2{globals_, "r2", "Kr", settings_.drmax(), settings_.kr()};
    VarDef z1{globals_, "z1", "Kz", settings_.zlength(), settings_.kz()};
    VarDef z2{globals_, "z2", "Kz", settings_.zlength(), settings_.kz()};

    //0.75 below comes from phi range for coordinate can be larger than for sector
    VarDef phi1{globals_, "phi1", "Kphi", settings_.dphisector() / 0.75, settings_.kphi1()};
    VarDef phi2{globals_, "phi2", "Kphi", settings_.dphisector() / 0.75, settings_.kphi1()};

    VarDef rproj0{globals_, "rproj0", "Kr", settings_.rmax(N_LAYER - 1), settings_.kr()};
    VarDef rproj1{globals_, "rproj1", "Kr", settings_.rmax(N_LAYER - 1), settings_.kr()};
    VarDef rproj2{globals_, "rproj2", "Kr", settings_.rmax(N_LAYER - 1), settings_.kr()};
    VarDef rproj3{globals_, "rproj3", "Kr", settings_.rmax(N_LAYER - 1), settings_.kr()};

    VarDef zproj0{globals_, "zproj0", "Kz", settings_.zmax(N_DISK - 1), settings_.kz()};
    VarDef zproj1{globals_, "zproj1", "Kz", settings_.zmax(N_DISK - 1), settings_.kz()};
    VarDef zproj2{globals_, "zproj2", "Kz", settings_.zmax(N_DISK - 1), settings_.kz()};
    VarDef zproj3{globals_, "zproj3", "Kz", settings_.zmax(N_DISK - 1), settings_.kz()};
    VarDef zproj4{globals_, "zproj4", "Kz", settings_.zmax(N_DISK - 1), settings_.kz()};

    //calculations

    //tracklet
    VarAdd r1abs{globals_, "r1abs", &r1, &r1mean, settings_.rmax(N_LAYER - 1)};
    VarAdd r2abs{globals_, "r2abs", &r2, &r2mean, settings_.rmax(N_LAYER - 1)};

    VarSubtract dr{globals_, "dr", &r2, &r1};

    //R LUT
    VarInv drinv{globals_, "drinv", &dr, 0, 18, 24, 0, VarInv::mode::both};

    VarSubtract dphi{globals_, "dphi", &phi2, &phi1, settings_.dphisector() / 4.};
    VarSubtract dz{globals_, "dz", &z2, &z1, dz_max};

    VarMult delta0{globals_, "delta0", &dphi, &drinv, delta0_max};
    VarMult deltaZ{globals_, "deltaZ", &dz, &drinv};
    VarMult delta1{globals_, "delta1", &r1abs, &delta0};
    VarMult delta2{globals_, "delta2", &r2abs, &delta0};
    VarMult a2a{globals_, "a2a", &delta1, &delta2, a2a_max};
    VarNounits a2b{globals_, "a2b", &a2a};
    VarSubtract a2{globals_, "a2", &plus2, &a2b, a2_max};
    VarNeg a2n{globals_, "a2n", &a2};
    VarShift a{globals_, "a", &a2, 1};

    VarAdd Rabs{globals_, "Rabs", &r1abs, &r2abs};
    VarTimesC R6{globals_, "R6", &Rabs, 1. / 6., 12};

    VarMult x4{globals_, "x4", &R6, &delta0};
    VarMult x6a{globals_, "x6a", &delta2, &x4, 2 * x6a_max};
    VarNounits x6b{globals_, "x6b", &x6a};
    VarAdd x6m{globals_, "x6m", &minus1, &x6b, x6m_max};
    VarMult phi0a{globals_, "phi0a", &delta1, &x6m, settings_.dphisector()};

    VarMult z0a{globals_, "z0a", &r1abs, &deltaZ, settings_.zlength()};
    VarMult z0b{globals_, "z0b", &z0a, &x6m, settings_.zlength()};

    VarAdd phi0{globals_, "phi0", &phi1, &phi0a, 2 * settings_.dphisector()};
    VarMult rinv{globals_, "rinv", &a2n, &delta0, 2 * settings_.maxrinv()};
    VarMult t{globals_, "t", &a, &deltaZ, t_max};
    VarAdd z0{globals_, "z0", &z1, &z0b, 2 * z0_max};

    VarAdjustK rinv_final{
        globals_, "rinv_final", &rinv, settings_.kphi1() / settings_.kr() * pow(2, settings_.rinv_shift())};
    VarAdjustK phi0_final{globals_, "phi0_final", &phi0, settings_.kphi1() * pow(2, settings_.phi0_shift())};
    VarAdjustKR t_final{globals_, "t_final", &t, settings_.kz() / settings_.kr() * pow(2, settings_.t_shift())};
    VarAdjustKR z0_final{globals_, "z0_final", &z0, settings_.kz() * pow(2, settings_.z0_shift())};

    //projection to r
    VarShift x2{globals_, "x2", &delta0, 1};

    VarMult x1_0{globals_, "x1_0", &x2, &rproj0};
    VarMult x1_1{globals_, "x1_1", &x2, &rproj1};
    VarMult x1_2{globals_, "x1_2", &x2, &rproj2};
    VarMult x1_3{globals_, "x1_3", &x2, &rproj3};

    VarMult x8_0{globals_, "x8_0", &x1_0, &a2n, x8_max};
    VarMult x8_1{globals_, "x8_1", &x1_1, &a2n, x8_max};
    VarMult x8_2{globals_, "x8_2", &x1_2, &a2n, x8_max};
    VarMult x8_3{globals_, "x8_3", &x1_3, &a2n, x8_max};

    VarMult x12_0{globals_, "x12_0", &x8_0, &x8_0};
    VarMult x12_1{globals_, "x12_1", &x8_1, &x8_1};
    VarMult x12_2{globals_, "x12_2", &x8_2, &x8_2};
    VarMult x12_3{globals_, "x12_3", &x8_3, &x8_3};

    VarNounits x12A_0{globals_, "x12A_0", &x12_0};
    VarNounits x12A_1{globals_, "x12A_1", &x12_1};
    VarNounits x12A_2{globals_, "x12A_2", &x12_2};
    VarNounits x12A_3{globals_, "x12A_3", &x12_3};

    VarTimesC x20_0{globals_, "x20_0", &x12A_0, 1. / 6.};
    VarTimesC x20_1{globals_, "x20_1", &x12A_1, 1. / 6.};
    VarTimesC x20_2{globals_, "x20_2", &x12A_2, 1. / 6.};
    VarTimesC x20_3{globals_, "x20_3", &x12A_3, 1. / 6.};

    VarAdd x10_0{globals_, "x10_0", &plus1, &x20_0};
    VarAdd x10_1{globals_, "x10_1", &plus1, &x20_1};
    VarAdd x10_2{globals_, "x10_2", &plus1, &x20_2};
    VarAdd x10_3{globals_, "x10_3", &plus1, &x20_3};

    VarMult x22_0{globals_, "x22_0", &x8_0, &x10_0, 2 * x22_max};
    VarMult x22_1{globals_, "x22_1", &x8_1, &x10_1, 2 * x22_max};
    VarMult x22_2{globals_, "x22_2", &x8_2, &x10_2, 2 * x22_max};
    VarMult x22_3{globals_, "x22_3", &x8_3, &x10_3, 2 * x22_max};

    VarSubtract phiL_0{globals_, "phiL_0", &phi0_final, &x22_0, -1, phi0_final.nbits() + 1};
    VarSubtract phiL_1{globals_, "phiL_1", &phi0_final, &x22_1, -1, phi0_final.nbits() + 1};
    VarSubtract phiL_2{globals_, "phiL_2", &phi0_final, &x22_2, -1, phi0_final.nbits() + 1};
    VarSubtract phiL_3{globals_, "phiL_3", &phi0_final, &x22_3, -1, phi0_final.nbits() + 1};

    VarShift x3{globals_, "x3", &rinv, 1};
    VarNeg der_phiL{globals_, "der_phiL", &x3};

    VarAdjustK phiL_0_final{globals_, "phiL_0_final", &phiL_0, settings_.kphi1() * pow(2, settings_.SS_phiL_shift())};
    VarAdjustK phiL_1_final{globals_, "phiL_1_final", &phiL_1, settings_.kphi1() * pow(2, settings_.SS_phiL_shift())};
    VarAdjustK phiL_2_final{globals_, "phiL_2_final", &phiL_2, settings_.kphi1() * pow(2, settings_.SS_phiL_shift())};
    VarAdjustK phiL_3_final{globals_, "phiL_3_final", &phiL_3, settings_.kphi1() * pow(2, settings_.SS_phiL_shift())};

    VarAdjustK der_phiL_final{globals_,
                              "der_phiL_final",
                              &der_phiL,
                              settings_.kphi1() / settings_.kr() * pow(2, settings_.SS_phiderL_shift())};

    VarMult x11_0{globals_, "x11_0", &rproj0, &t};
    VarMult x11_1{globals_, "x11_1", &rproj1, &t};
    VarMult x11_2{globals_, "x11_2", &rproj2, &t};
    VarMult x11_3{globals_, "x11_3", &rproj3, &t};

    VarMult x23_0{globals_, "x23_0", &x11_0, &x10_0, 2 * x23_max};
    VarMult x23_1{globals_, "x23_1", &x11_1, &x10_1, 2 * x23_max};
    VarMult x23_2{globals_, "x23_2", &x11_2, &x10_2, 2 * x23_max};
    VarMult x23_3{globals_, "x23_3", &x11_3, &x10_3, 2 * x23_max};

    VarAdd zL_0{globals_, "zL_0", &z0, &x23_0};
    VarAdd zL_1{globals_, "zL_1", &z0, &x23_1};
    VarAdd zL_2{globals_, "zL_2", &z0, &x23_2};
    VarAdd zL_3{globals_, "zL_3", &z0, &x23_3};

    VarAdjustKR zL_0_final{globals_, "zL_0_final", &zL_0, settings_.kz() * pow(2, settings_.PS_zL_shift())};
    VarAdjustKR zL_1_final{globals_, "zL_1_final", &zL_1, settings_.kz() * pow(2, settings_.PS_zL_shift())};
    VarAdjustKR zL_2_final{globals_, "zL_2_final", &zL_2, settings_.kz() * pow(2, settings_.PS_zL_shift())};
    VarAdjustKR zL_3_final{globals_, "zL_3_final", &zL_3, settings_.kz() * pow(2, settings_.PS_zL_shift())};

    VarAdjustK der_zL_final{
        globals_, "der_zL_final", &t_final, settings_.kz() / settings_.kr() * pow(2, settings_.PS_zderL_shift())};

    //projection to z
    //
    VarInv invt{globals_, "invt", &t_final, 0., 18, 26, 1, VarInv::mode::both, 13};

    VarMult x7{globals_, "x7", &x2, &a2};

    VarSubtract x5_0{globals_, "x5_0", &zproj0, &z0};
    VarSubtract x5_1{globals_, "x5_1", &zproj1, &z0};
    VarSubtract x5_2{globals_, "x5_2", &zproj2, &z0};
    VarSubtract x5_3{globals_, "x5_3", &zproj3, &z0};
    VarSubtract x5_4{globals_, "x5_4", &zproj4, &z0};

    VarMult x13_0{globals_, "x13_0", &x5_0, &invt, x13_max};
    VarMult x13_1{globals_, "x13_1", &x5_1, &invt, x13_max};
    VarMult x13_2{globals_, "x13_2", &x5_2, &invt, x13_max};
    VarMult x13_3{globals_, "x13_3", &x5_3, &invt, x13_max};
    VarMult x13_4{globals_, "x13_4", &x5_4, &invt, x13_max};

    VarMult x25_0{globals_, "x25_0", &x13_0, &x7, 2 * settings_.dphisector()};
    VarMult x25_1{globals_, "x25_1", &x13_1, &x7, 2 * settings_.dphisector()};
    VarMult x25_2{globals_, "x25_2", &x13_2, &x7, 2 * settings_.dphisector()};
    VarMult x25_3{globals_, "x25_3", &x13_3, &x7, 2 * settings_.dphisector()};
    VarMult x25_4{globals_, "x25_4", &x13_4, &x7, 2 * settings_.dphisector()};

    VarAdd phiD_0{globals_, "phiD_0", &phi0, &x25_0, 2 * settings_.dphisector()};
    VarAdd phiD_1{globals_, "phiD_1", &phi0, &x25_1, 2 * settings_.dphisector()};
    VarAdd phiD_2{globals_, "phiD_2", &phi0, &x25_2, 2 * settings_.dphisector()};
    VarAdd phiD_3{globals_, "phiD_3", &phi0, &x25_3, 2 * settings_.dphisector()};
    VarAdd phiD_4{globals_, "phiD_4", &phi0, &x25_4, 2 * settings_.dphisector()};

    VarAdjustK phiD_0_final{globals_, "phiD_0_final", &phiD_0, settings_.kphi1() * pow(2, settings_.SS_phiD_shift())};
    VarAdjustK phiD_1_final{globals_, "phiD_1_final", &phiD_1, settings_.kphi1() * pow(2, settings_.SS_phiD_shift())};
    VarAdjustK phiD_2_final{globals_, "phiD_2_final", &phiD_2, settings_.kphi1() * pow(2, settings_.SS_phiD_shift())};
    VarAdjustK phiD_3_final{globals_, "phiD_3_final", &phiD_3, settings_.kphi1() * pow(2, settings_.SS_phiD_shift())};
    VarAdjustK phiD_4_final{globals_, "phiD_4_final", &phiD_4, settings_.kphi1() * pow(2, settings_.SS_phiD_shift())};

    VarMult der_phiD{globals_, "der_phiD", &x7, &invt, 4 * der_phiD_max};

    VarAdjustK der_phiD_final{globals_,
                              "der_phiD_final",
                              &der_phiD,
                              settings_.kphi1() / settings_.kr() * pow(2, settings_.SS_phiderD_shift())};

    VarMult x26_0{globals_, "x26_0", &x25_0, &x25_0};
    VarMult x26_1{globals_, "x26_1", &x25_1, &x25_1};
    VarMult x26_2{globals_, "x26_2", &x25_2, &x25_2};
    VarMult x26_3{globals_, "x26_3", &x25_3, &x25_3};
    VarMult x26_4{globals_, "x26_4", &x25_4, &x25_4};

    VarNounits x26A_0{globals_, "x26A_0", &x26_0};
    VarNounits x26A_1{globals_, "x26A_1", &x26_1};
    VarNounits x26A_2{globals_, "x26A_2", &x26_2};
    VarNounits x26A_3{globals_, "x26A_3", &x26_3};
    VarNounits x26A_4{globals_, "x26A_4", &x26_4};

    VarTimesC x9_0{globals_, "x9_0", &x26A_0, 1. / 6.};
    VarTimesC x9_1{globals_, "x9_1", &x26A_1, 1. / 6.};
    VarTimesC x9_2{globals_, "x9_2", &x26A_2, 1. / 6.};
    VarTimesC x9_3{globals_, "x9_3", &x26A_3, 1. / 6.};
    VarTimesC x9_4{globals_, "x9_4", &x26A_4, 1. / 6.};

    VarSubtract x27m_0{globals_, "x27_0", &plus1, &x9_0};
    VarSubtract x27m_1{globals_, "x27_1", &plus1, &x9_1};
    VarSubtract x27m_2{globals_, "x27_2", &plus1, &x9_2};
    VarSubtract x27m_3{globals_, "x27_3", &plus1, &x9_3};
    VarSubtract x27m_4{globals_, "x27_4", &plus1, &x9_4};

    VarMult rD_0{globals_, "rD_0", &x13_0, &x27m_0, 2 * settings_.rmaxdisk()};
    VarMult rD_1{globals_, "rD_1", &x13_1, &x27m_1, 2 * settings_.rmaxdisk()};
    VarMult rD_2{globals_, "rD_2", &x13_2, &x27m_2, 2 * settings_.rmaxdisk()};
    VarMult rD_3{globals_, "rD_3", &x13_3, &x27m_3, 2 * settings_.rmaxdisk()};
    VarMult rD_4{globals_, "rD_4", &x13_4, &x27m_4, 2 * settings_.rmaxdisk()};

    VarAdjustK rD_0_final{globals_, "rD_0_final", &rD_0, settings_.kr() * pow(2, settings_.PS_rD_shift())};
    VarAdjustK rD_1_final{globals_, "rD_1_final", &rD_1, settings_.kr() * pow(2, settings_.PS_rD_shift())};
    VarAdjustK rD_2_final{globals_, "rD_2_final", &rD_2, settings_.kr() * pow(2, settings_.PS_rD_shift())};
    VarAdjustK rD_3_final{globals_, "rD_3_final", &rD_3, settings_.kr() * pow(2, settings_.PS_rD_shift())};
    VarAdjustK rD_4_final{globals_, "rD_4_final", &rD_4, settings_.kr() * pow(2, settings_.PS_rD_shift())};

    VarAdjustK der_rD_final{
        globals_, "der_rD_final", &invt, settings_.kr() / settings_.kz() * pow(2, settings_.PS_rderD_shift())};

    VarCut rinv_final_cut{globals_, &rinv_final, -settings_.rinvcut(), settings_.rinvcut()};
    // the following two are not associated with any variable yet; this is done
    // in the constructor of this class since it depends on the layer
    VarCut z0_final_L1_cut{globals_, -settings_.z0cut(), settings_.z0cut()};
    VarCut z0_final_cut{globals_, -1.5 * settings_.z0cut(), 1.5 * settings_.z0cut()};

    VarCut r1abs_cut{globals_, &r1abs, -settings_.rmax(5), settings_.rmax(5)};
    VarCut r2abs_cut{globals_, &r2abs, -settings_.rmax(5), settings_.rmax(5)};
    VarCut dphi_cut{globals_, &dphi, -settings_.dphisector() / 4., settings_.dphisector() / 4.};
    VarCut dz_cut{globals_, &dz, -dz_max, dz_max};
    VarCut delta0_cut{globals_, &delta0, -delta0_max, delta0_max};
    VarCut a2a_cut{globals_, &a2a, -a2a_max, a2a_max};
    VarCut a2_cut{globals_, &a2, -a2_max, a2_max};
    VarCut x6a_cut{globals_, &x6a, -x6a_max, x6a_max};
    VarCut x6m_cut{globals_, &x6m, -x6m_max, x6m_max};
    VarCut phi0a_cut{globals_, &phi0a, -settings_.dphisector(), settings_.dphisector()};
    VarCut z0a_cut{globals_, &z0a, (-1) * settings_.zlength(), settings_.zlength()};
    VarCut phi0_cut{globals_, &phi0, -2 * settings_.dphisector(), 2 * settings_.dphisector()};
    VarCut rinv_cut{globals_, &rinv, -settings_.maxrinv(), settings_.maxrinv()};
    VarCut t_cut{globals_, &t, -t_max, t_max};
    VarCut z0_cut{globals_, &z0, -z0_max, z0_max};
    VarCut x8_0_cut{globals_, &x8_0, -x8_max, x8_max};
    VarCut x8_1_cut{globals_, &x8_1, -x8_max, x8_max};
    VarCut x8_2_cut{globals_, &x8_2, -x8_max, x8_max};
    VarCut x8_3_cut{globals_, &x8_3, -x8_max, x8_max};
    VarCut x22_0_cut{globals_, &x22_0, -x22_max, x22_max};
    VarCut x22_1_cut{globals_, &x22_1, -x22_max, x22_max};
    VarCut x22_2_cut{globals_, &x22_2, -x22_max, x22_max};
    VarCut x22_3_cut{globals_, &x22_3, -x22_max, x22_max};
    VarCut x23_0_cut{globals_, &x23_0, -x23_max, x23_max};
    VarCut x23_1_cut{globals_, &x23_1, -x23_max, x23_max};
    VarCut x23_2_cut{globals_, &x23_2, -x23_max, x23_max};
    VarCut x23_3_cut{globals_, &x23_3, -x23_max, x23_max};
    VarCut x13_0_cut{globals_, &x13_0, -x13_max, x13_max};
    VarCut x13_1_cut{globals_, &x13_1, -x13_max, x13_max};
    VarCut x13_2_cut{globals_, &x13_2, -x13_max, x13_max};
    VarCut x13_3_cut{globals_, &x13_3, -x13_max, x13_max};
    VarCut x13_4_cut{globals_, &x13_4, -x13_max, x13_max};
    VarCut x25_0_cut{globals_, &x25_0, -settings_.dphisector(), settings_.dphisector()};
    VarCut x25_1_cut{globals_, &x25_1, -settings_.dphisector(), settings_.dphisector()};
    VarCut x25_2_cut{globals_, &x25_2, -settings_.dphisector(), settings_.dphisector()};
    VarCut x25_3_cut{globals_, &x25_3, -settings_.dphisector(), settings_.dphisector()};
    VarCut x25_4_cut{globals_, &x25_4, -settings_.dphisector(), settings_.dphisector()};
    VarCut phiD_0_cut{globals_, &phiD_0, -2 * settings_.dphisector(), 2 * settings_.dphisector()};
    VarCut phiD_1_cut{globals_, &phiD_1, -2 * settings_.dphisector(), 2 * settings_.dphisector()};
    VarCut phiD_2_cut{globals_, &phiD_2, -2 * settings_.dphisector(), 2 * settings_.dphisector()};
    VarCut phiD_3_cut{globals_, &phiD_3, -2 * settings_.dphisector(), 2 * settings_.dphisector()};
    VarCut phiD_4_cut{globals_, &phiD_4, -2 * settings_.dphisector(), 2 * settings_.dphisector()};
    VarCut der_phiD_cut{globals_, &der_phiD, -der_phiD_max, der_phiD_max};
    VarCut rD_0_cut{globals_, &rD_0, -settings_.rmaxdisk(), settings_.rmaxdisk()};
    VarCut rD_1_cut{globals_, &rD_1, -settings_.rmaxdisk(), settings_.rmaxdisk()};
    VarCut rD_2_cut{globals_, &rD_2, -settings_.rmaxdisk(), settings_.rmaxdisk()};
    VarCut rD_3_cut{globals_, &rD_3, -settings_.rmaxdisk(), settings_.rmaxdisk()};
    VarCut rD_4_cut{globals_, &rD_4, -settings_.rmaxdisk(), settings_.rmaxdisk()};

    VarCut t_disk_cut_left{globals_, &t, -t_disk_max, -t_disk_min};
    VarCut t_disk_cut_right{globals_, &t, t_disk_min, t_disk_max};
    VarCut t_layer_cut{globals_, &t, -t_layer_max, t_layer_max};

    // the following flags are used to apply the cuts in TrackletCalculator
    // and in the output Verilog
    VarFlag valid_trackpar{globals_, "valid_trackpar", &rinv_final, &phi0_final, &t_final, &z0_final};

    VarFlag valid_phiL_0{globals_, "valid_phiL_0", &phiL_0_final};
    VarFlag valid_phiL_1{globals_, "valid_phiL_1", &phiL_1_final};
    VarFlag valid_phiL_2{globals_, "valid_phiL_2", &phiL_2_final};
    VarFlag valid_phiL_3{globals_, "valid_phiL_3", &phiL_3_final};

    VarFlag valid_zL_0{globals_, "valid_zL_0", &zL_0_final};
    VarFlag valid_zL_1{globals_, "valid_zL_1", &zL_1_final};
    VarFlag valid_zL_2{globals_, "valid_zL_2", &zL_2_final};
    VarFlag valid_zL_3{globals_, "valid_zL_3", &zL_3_final};

    VarFlag valid_der_phiL{globals_, "valid_der_phiL", &der_phiL_final};
    VarFlag valid_der_zL{globals_, "valid_der_zL", &der_zL_final};

    VarFlag valid_phiD_0{globals_, "valid_phiD_0", &phiD_0_final};
    VarFlag valid_phiD_1{globals_, "valid_phiD_1", &phiD_1_final};
    VarFlag valid_phiD_2{globals_, "valid_phiD_2", &phiD_2_final};
    VarFlag valid_phiD_3{globals_, "valid_phiD_3", &phiD_3_final};
    VarFlag valid_phiD_4{globals_, "valid_phiD_4", &phiD_4_final};

    VarFlag valid_rD_0{globals_, "valid_rD_0", &rD_0_final};
    VarFlag valid_rD_1{globals_, "valid_rD_1", &rD_1_final};
    VarFlag valid_rD_2{globals_, "valid_rD_2", &rD_2_final};
    VarFlag valid_rD_3{globals_, "valid_rD_3", &rD_3_final};
    VarFlag valid_rD_4{globals_, "valid_rD_4", &rD_4_final};

    VarFlag valid_der_phiD{globals_, "valid_der_phiD", &der_phiD_final};
    VarFlag valid_der_rD{globals_, "valid_der_rD", &der_rD_final};
  };
};  // namespace trklet

#endif
