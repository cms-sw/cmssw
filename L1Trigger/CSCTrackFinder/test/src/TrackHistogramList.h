
#ifndef jhugon_TrackHistogramList_h
#define jhugon_TrackHistogramList_h
// system include files
#include <vector>
#include <string>
// user include files
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include <TF1.h>
#include <TH2.h>
namespace csctf_analysis {
  class TrackHistogramList {
  public:
    TrackHistogramList(const std::string dirname, const edm::ParameterSet *parameters);
    TH2F *SimPt_vs_TFPt_FWD;
    TH2F *SimPt_vs_TFPt_DT;
    TH1F *matchedRefPt_FWD;
    TH1F *matchedRefPt_DT;
    TH1F *modeOcc, *modeOccDT, *modeOccCSCOnly, *modeOccOverlap, *modeOccHighEta;
    TH1F *BX;
    TH1F *FR;
    TH1F *Eta, *signedEta, *Phi, *Phi_mod_10_endcap1, *Phi_mod_10_endcap2, *Pt, *Pz, *P, *Quality, *Radius;  //,*EHalo;
    TH1F *matchPhi, *matchPhi_mod_10_Q3_endcap1, *matchPhi_mod_10_Q2_endcap1, *matchPhi_mod_10_Q3_endcap2,
        *matchPhi_mod_10_Q2_endcap2;
    TH1F *matchEta, *signedMatchEta, *matchPt, *matchRadius;  //, *HaloPRes;
    TH1F *EtaQ3, *EtaQ2, *EtaQ1;
    TH1F *signedEtaQ3, *signedEtaQ2, *signedEtaQ1;
    TH1F *PhiQ3, *PhiQ2, *PhiQ1;
    TH1F *PtQ3, *PtQ2, *PtQ1;
    TH1F *ghostPhi, *ghostEta, *ghostSignedEta, *ghostPt, *ghostRadius;  //, *ghostPhiSector;
    TH1F *ghostEtaQ3, *ghostEtaQ2, *ghostEtaQ1;
    TH1F *ghostSignedEtaQ3, *ghostSignedEtaQ2, *ghostSignedEtaQ1;
    TH1F *ghostPhiQ3, *ghostPhiQ2, *ghostPhiQ1;
    TH1F *ghostPtQ3, *ghostPtQ2, *ghostPtQ1;
    TH1F *ghostQuality;  //, *ghostPhiSectorTrk;
    TH1F *matchTFPt10Overall, *matchTFPt12Overall, *matchTFPt16Overall, *matchTFPt20Overall, *matchTFPt40Overall,
        *matchTFPt60Overall;
    TH1F *matchTFPt10CSCOnly, *matchTFPt12CSCOnly, *matchTFPt16CSCOnly, *matchTFPt20CSCOnly, *matchTFPt40CSCOnly,
        *matchTFPt60CSCOnly;
    TH1F *matchTFPt10CSCRestricted, *matchTFPt12CSCRestricted, *matchTFPt16CSCRestricted, *matchTFPt20CSCRestricted,
        *matchTFPt40CSCRestricted, *matchTFPt60CSCRestricted;
    TH1F *matchTFPt10Overlap, *matchTFPt12Overlap, *matchTFPt16Overlap, *matchTFPt20Overlap, *matchTFPt40Overlap,
        *matchTFPt60Overlap;
    TH1F *matchTFPt10HighEta, *matchTFPt12HighEta, *matchTFPt16HighEta, *matchTFPt20HighEta, *matchTFPt40HighEta,
        *matchTFPt60HighEta;
    TH1F *matchTFPt10DTOnly, *matchTFPt12DTOnly, *matchTFPt16DTOnly, *matchTFPt20DTOnly, *matchTFPt40DTOnly,
        *matchTFPt60DTOnly;
    TH1F *matchPtCSCOnly, *matchPtOverlap, *matchPtHighEta, *matchPtOverall, *matchPtDTOnly, *matchPtCSCRestricted;
    TH1F *matchMode;
    TH1F *fidPtDen, *ptDenOverall, *ptDenCSCOnly, *ptDenOverlap, *ptDenHighEta, *ptDenDTOnly, *ptDenCSCRestricted;
    TH1F *rateHist;
    double getPtStep() { return ptStep; };

  private:
    edm::Service<TFileService> fs;
    double ptStep;
  };
}  // namespace csctf_analysis
#endif
