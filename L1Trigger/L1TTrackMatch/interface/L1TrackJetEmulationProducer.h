#ifndef L1Trigger_L1TTrackMatch_L1TrackJetEmulationProducer_HH
#define L1Trigger_L1TTrackMatch_L1TrackJetEmulationProducer_HH
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <string>
#include <cstdlib>
#include "DataFormats/L1Trigger/interface/TkJetWord.h"

//For precision studies
const int PT_EXTRABITS = 0;
const int ETA_EXTRABITS = 0;
const int PHI_EXTRABITS = 0;
const int Z0_EXTRABITS = 0;

typedef ap_ufixed<16 + PT_EXTRABITS, 11, AP_TRN, AP_SAT> pt_intern;
typedef ap_int<14 + ETA_EXTRABITS> glbeta_intern;
typedef ap_int<14 + PHI_EXTRABITS> glbphi_intern;
typedef ap_int<10 + Z0_EXTRABITS> z0_intern;  // 40cm / 0.1

namespace convert {
  const int INTPHI_PI = 720;
  const int INTPHI_TWOPI = 2 * INTPHI_PI;
  static const float INTPT_LSB_POW = pow(2.0, -5 - PT_EXTRABITS);
  static const float INTPT_LSB = INTPT_LSB_POW;
  static const float ETA_LSB_POW = pow(2.0, -1 * ETA_EXTRABITS);
  static const float ETA_LSB = M_PI / pow(2.0, 12) * ETA_LSB_POW;
  static const float PHI_LSB_POW = pow(2.0, -1 * PHI_EXTRABITS);
  static const float PHI_LSB = M_PI / pow(2.0, 12) * PHI_LSB_POW;
  static const float Z0_LSB_POW = pow(2.0, -1 * Z0_EXTRABITS);
  static const float Z0_LSB = 0.05 * Z0_LSB_POW;
  inline float floatPt(pt_intern pt) { return pt.to_float(); }
  inline int intPt(pt_intern pt) { return (ap_ufixed<18 + PT_EXTRABITS, 13 + PT_EXTRABITS>(pt)).to_int(); }
  inline float floatEta(glbeta_intern eta) { return eta.to_float() * ETA_LSB; }
  inline float floatPhi(glbphi_intern phi) { return phi.to_float() * PHI_LSB; }
  inline float floatZ0(z0_intern z0) { return z0.to_float() * Z0_LSB; }

  inline pt_intern makePt(int pt) { return ap_ufixed<18 + PT_EXTRABITS, 13 + PT_EXTRABITS>(pt); }
  inline pt_intern makePtFromFloat(float pt) { return pt_intern(INTPT_LSB_POW * round(pt / INTPT_LSB_POW)); }
  inline z0_intern makeZ0(float z0) { return z0_intern(round(z0 / Z0_LSB)); }

  inline ap_uint<pt_intern::width> ptToInt(pt_intern pt) {
    // note: this can be synthethized, e.g. when pT is used as intex in a LUT
    ap_uint<pt_intern::width> ret = 0;
    ret(pt_intern::width - 1, 0) = pt(pt_intern::width - 1, 0);
    return ret;
  }

  inline glbeta_intern makeGlbEta(float eta) { return round(eta / ETA_LSB); }
  inline glbeta_intern makeGlbEtaRoundEven(float eta) {
    glbeta_intern ghweta = round(eta / ETA_LSB);
    return (ghweta % 2) ? glbeta_intern(ghweta + 1) : ghweta;
  }

  inline glbphi_intern makeGlbPhi(float phi) { return round(phi / PHI_LSB); }

};  // namespace convert

//Each individual box in the eta and phi dimension.
//  Also used to store final cluster data for each zbin.
struct TrackJetEmulationEtaPhiBin {
  pt_intern pTtot;
  l1t::TkJetWord::nt_t ntracks;
  l1t::TkJetWord::nx_t nxtracks;
  bool used;
  glbphi_intern phi;  //average phi value (halfway b/t min and max)
  glbeta_intern eta;  //average eta value
};

//store important information for plots
struct TrackJetEmulationMaxZBin {
  int znum = 0;    //Numbered from 0 to nzbins (16, 32, or 64) in order
  int nclust = 0;  //number of clusters in this bin
  z0_intern zbincenter = 0;
  TrackJetEmulationEtaPhiBin *clusters = nullptr;  //list of all the clusters in this bin
  pt_intern ht = 0;                                //sum of all cluster pTs--only the zbin with the maximum ht is stored
};
#endif
