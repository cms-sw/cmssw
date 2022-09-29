
#ifndef jhugon_EffHistogramList_h
#define jhugon_EffHistogramList_h
// system include files
#include <vector>
#include <string>
#include <fstream>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include <TCanvas.h>
#include <TStyle.h>
#include <TLegend.h>
#include <TLatex.h>
#include <TF1.h>
#include <TH2.h>
#include "L1Trigger/CSCTrackFinder/test/src/TrackHistogramList.h"
#include <TMath.h>

namespace csctf_analysis {
  class EffHistogramList {
  public:
    EffHistogramList(const std::string dirname, const edm::ParameterSet *parameters);
    void ComputeEff(TrackHistogramList *);
    void Print();
    TH1F *EffPhi_mod_10_Q3_endcap1, *EffPhi_mod_10_Q2_endcap1;
    TH1F *EffPhi_mod_10_Q3_endcap2, *EffPhi_mod_10_Q2_endcap2;
    TH1F *modeOcc;
    TH1F *EffEtaAll, *EffEtaQ3, *EffEtaQ2, *EffEtaQ1;
    TH1F *EffSignedEtaAll, *EffSignedEtaQ3, *EffSignedEtaQ2, *EffSignedEtaQ1;
    TH1F *EffPhiQ3, *EffPhiQ2, *EffPhiQ1, *EffPhi;
    TH1F *EffPtOverall, *EffPtCSCOnly, *EffPtOverlap, *EffPtHighEta, *EffPtDTOnly, *EffPtCSCRestricted;
    TH1F *EffTFPt10Overall, *EffTFPt12Overall, *EffTFPt16Overall, *EffTFPt20Overall, *EffTFPt40Overall,
        *EffTFPt60Overall;
    TH1F *EffTFPt10CSCOnly, *EffTFPt12CSCOnly, *EffTFPt16CSCOnly, *EffTFPt20CSCOnly, *EffTFPt40CSCOnly,
        *EffTFPt60CSCOnly;
    TH1F *EffTFPt10CSCRestricted, *EffTFPt12CSCRestricted, *EffTFPt16CSCRestricted, *EffTFPt20CSCRestricted,
        *EffTFPt40CSCRestricted, *EffTFPt60CSCRestricted;
    TH1F *EffTFPt10DTOnly, *EffTFPt12DTOnly, *EffTFPt16DTOnly, *EffTFPt20DTOnly, *EffTFPt40DTOnly, *EffTFPt60DTOnly;
    TH1F *EffTFPt10Overlap, *EffTFPt12Overlap, *EffTFPt16Overlap, *EffTFPt20Overlap, *EffTFPt40Overlap,
        *EffTFPt60Overlap;
    TH1F *EffTFPt10HighEta, *EffTFPt12HighEta, *EffTFPt16HighEta, *EffTFPt20HighEta, *EffTFPt40HighEta,
        *EffTFPt60HighEta;
    TLegend *TrackerLeg1, *TrackerLeg2, *TrackerLeg3;
    TLegend *TrackerLeg1CSCOnly, *TrackerLeg1Overlap, *TrackerLeg1HighEta, *TrackerLeg1DTOnly,
        *TrackerLeg1CSCRestricted, *TrackerLeg1Overall;
    TCanvas *PhiEff, *EtaEff, *SignedEtaEff, *PtEffAllOverall, *PtEffAllCSCOnly, *PtEffAllOverlap, *PtEffAllHighEta,
        *PtEffAllDTOnly, *PtEffAllCSCRestricted;
    TF1 *fitThreshOverall, *fitThreshCSCOnly, *fitThreshDTOnly, *fitThreshCSCRestricted, *fitThreshOverlap,
        *fitThreshHighEta;

  private:
    edm::Service<TFileService> fs;
    std::string PtEffStatsFilename;
    void DrawPtEffHists(std::string region,
                        TCanvas *canvas,
                        TF1 *fit,
                        TLegend *legend,
                        std::vector<std::string> thresholds,
                        std::vector<TH1F *> PtEffHists);
    void computeErrors(TrackHistogramList *);
    void divideHistograms(TrackHistogramList *);
    void computePtPlateauEff(std::ofstream *PtStats,
                             std::vector<double> PlateauDefinitions,
                             std::vector<std::string> thresholds,
                             std::vector<TH1F *> PtEffHists);
    TLatex *latexDescription;
  };
  Double_t thresh(Double_t *pt, Double_t *par);
}  // namespace csctf_analysis
#endif
