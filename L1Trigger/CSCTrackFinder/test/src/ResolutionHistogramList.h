
#ifndef jhugon_ResolutionHistogramList_h
#define jhugon_ResolutionHistogramList_h

// system include files
#include <vector>
#include <string>

// user include files

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "L1Trigger/CSCTrackFinder/test/src/TFTrack.h"
#include "L1Trigger/CSCTrackFinder/test/src/RefTrack.h"

namespace csctf_analysis {
  class ResolutionHistogramList {
  public:
    ResolutionHistogramList(const std::string dirname, const edm::ParameterSet *parameters);

    TH1F *PtQ1Res, *PtQ2Res, *PtQ3Res;
    TH1F *PhiQ1Res, *PhiQ2Res, *PhiQ3Res;
    TH1F *EtaQ1Res, *EtaQ2Res, *EtaQ3Res;

    TH1F *PtQ2ResGolden, *PhiQ2ResGolden;
    TH1F *PtQ2ResHighEta, *PhiQ2ResHighEta;
    TH1F *PtQ2ResOverlap, *PhiQ2ResOverlap;

    void FillResolutionHist(RefTrack refTrk, TFTrack tfTrk);
    void Print();

  private:
    edm::Service<TFileService> fs;
  };
}  // namespace csctf_analysis
#endif
