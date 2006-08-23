#ifndef CSCTrackFinder_CSCTFPtLUT_h
#define CSCTrackFinder_CSCTFPtLUT_h

#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <L1Trigger/CSCTrackFinder/interface/CSCTrackFinderDataTypes.h>
#include <DataFormats/L1GlobalMuonTrigger/interface/L1MuTriggerScales.h>
#include <L1Trigger/CSCTrackFinder/interface/CSCTFPtMethods.h>

class CSCTFPtLUT
{
 public:
  CSCTFPtLUT(const edm::ParameterSet&);
  CSCTFPtLUT(const CSCTFPtLUT&);
  ~CSCTFPtLUT() { if(pt_lut) delete pt_lut; pt_lut = NULL; }

  CSCTFPtLUT& operator=(const CSCTFPtLUT&);

  ptdat Pt(const ptadd&) const;

  ptdat Pt(const unsigned&) const;

  ptdat Pt(const unsigned& delta_phi_12, const unsigned& delta_phi23,
	   const unsigned& track_eta, const unsigned& track_mode,
	   const unsigned& track_fr, const unsigned& delta_phi_sign) const;

  ptdat Pt(const unsigned& delta_phi_12, const unsigned& track_eta, 
	   const unsigned& track_mode, const unsigned& track_fr, 
	   const unsigned& delta_phi_sign) const;

 private:
  static ptdat* pt_lut;
  static bool lut_read_in;
  static L1MuTriggerScales trigger_scale;
  static CSCTFPtMethods ptMethods;
  
  bool read_pt_lut;
  unsigned pt_method, lowQualityFlag;
  

  ptdat calcPt(const ptadd&) const;
  unsigned trackQuality(const unsigned& eta, const unsigned& mode) const;
  void readLUT();
};

#endif
