#ifndef CSCTrackFinder_CSCTFPtLUT_h
#define CSCTrackFinder_CSCTFPtLUT_h

#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <L1Trigger/CSCTrackFinder/interface/CSCTrackFinderDataTypes.h>
#include <CondFormats/L1TObjects/interface/L1MuTriggerScales.h>
#include <CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h>
#include <L1Trigger/CSCTrackFinder/interface/CSCTFPtMethods.h>
#include <FWCore/ParameterSet/interface/FileInPath.h>
///KK
#include <FWCore/Framework/interface/EventSetup.h>
///

class CSCTFPtLUT {
public:
  ///KK
  CSCTFPtLUT(const edm::EventSetup& c);
  ///

  CSCTFPtLUT(const edm::ParameterSet&, const L1MuTriggerScales* scales, const L1MuTriggerPtScale* ptScale);

  CSCTFPtLUT(const CSCTFPtLUT&);
  ~CSCTFPtLUT() {
    if (pt_lut)
      delete[] pt_lut;
    pt_lut = nullptr;
  }

  CSCTFPtLUT& operator=(const CSCTFPtLUT&);

  ptdat Pt(const ptadd&) const;

  ptdat Pt(const unsigned&) const;

  ptdat Pt(const unsigned& delta_phi_12,
           const unsigned& delta_phi23,
           const unsigned& track_eta,
           const unsigned& track_mode,
           const unsigned& track_fr,
           const unsigned& delta_phi_sign) const;

  ptdat Pt(const unsigned& delta_phi_12,
           const unsigned& track_eta,
           const unsigned& track_mode,
           const unsigned& track_fr,
           const unsigned& delta_phi_sign) const;

  static const int dPhiNLBMap_5bit[32];
  static const int dPhiNLBMap_7bit[128];
  static const int dPhiNLBMap_8bit[256];

  static const int dEtaCut_Low[24];
  static const int dEtaCut_Mid[24];
  static const int dEtaCut_High_A[24];
  static const int dEtaCut_High_B[24];
  static const int dEtaCut_High_C[24];
  static const int dEtaCut_Open[24];

  static const int getPtbyMLH;

private:
  static ptdat* pt_lut;
  static bool lut_read_in;
  const L1MuTriggerScales* trigger_scale;
  const L1MuTriggerPtScale* trigger_ptscale;
  CSCTFPtMethods ptMethods;

  bool read_pt_lut, isBinary, isBeamStartConf;
  edm::FileInPath pt_lut_file;
  unsigned pt_method, lowQualityFlag;

  ptdat calcPt(const ptadd&) const;
  //unsigned trackQuality(const unsigned& eta, const unsigned& mode) const;
  unsigned trackQuality(const unsigned& eta, const unsigned& mode, const unsigned& fr) const;
  void readLUT();
};

#endif
