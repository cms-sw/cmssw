
#ifndef jhugon_MultiplicityHistogramList_h
#define jhugon_MultiplicityHistogramList_h
// system include files
#include <vector>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "L1Trigger/CSCTrackFinder/test/src/TFTrack.h"

#include <TCanvas.h>

#include <TStyle.h>
#include <TLegend.h>
#include <TF1.h>
#include <TH2.h>
#include <TH1F.h>

namespace csctf_analysis {
  class MultiplicityHistogramList {
  public:
    MultiplicityHistogramList();

    void FillMultiplicityHist(std::vector<TFTrack> *);

    TH1F *nTFTracks, *highestTFPt, *highestTFPtMed, *highestTFPtLow;

  private:
    edm::Service<TFileService> fs;
  };
}  // namespace csctf_analysis
#endif
