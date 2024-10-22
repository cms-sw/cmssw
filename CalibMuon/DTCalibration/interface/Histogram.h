#ifndef CalibMuon_DTCalibration_Histogram_H
#define CalibMuon_DTCalibration_Histogram_H

#include "TH1.h"
#include "TH2.h"
#include "TString.h"

namespace dtcalibration {
  struct Histograms {
    TH1F* hT123Bad;
    TH1F* hT123LRL;
    TH1F* hT123LLR;
    TH1F* hT123LRR;
    TH1F* hT124Bad;
    TH1F* hT124LRR1gt4;
    TH1F* hT124LRR1lt4;
    TH1F* hT124LLR;
    TH1F* hT124LLLR;
    TH1F* hT124LLLL;
    TH1F* hT124LRLL;
    TH1F* hT124LRLR;
    TH1F* hT134Bad;
    TH1F* hT134LLR1gt4;
    TH1F* hT134LLR1lt4;
    TH1F* hT134LRR;
    TH1F* hT134LRLR;
    TH1F* hT134LRLL;
    TH1F* hT134LLLL;
    TH1F* hT134LLLR;
    TH1F* hT234Bad;
    TH1F* hT234LRL;
    TH1F* hT234LLR;
    TH1F* hT234LRR;

    void bookHistos() {
      hT123LRL = new TH1F("hT123LRL", "Tmax123 LRL", 2000, -1000., 1000.);
      hT123LLR = new TH1F("hT123LLR", "Tmax123 LLR", 2000, -1000., 1000.);
      hT123LRR = new TH1F("hT123LRR", "Tmax123 LRR", 2000, -1000., 1000.);
      hT123Bad = new TH1F("hT123Bad", "Bad Tmax123", 10, -5., 5.);
      hT124LRR1gt4 = new TH1F("hT124LRR1gt4", "Tmax124 LRR x1>x4", 2000, -1000., 1000.);
      hT124LRR1lt4 = new TH1F("hT124LRR1lt4", "Tmax124 LRR x1<x4", 2000, -1000., 1000.);
      hT124LLR = new TH1F("hT124LLR", "Tmax124 LLR", 2000, -1000., 1000.);
      hT124LLLR = new TH1F("hT124LLLR", "Tmax124 LLL Dir.R", 2000, -1000., 1000.);
      hT124LLLL = new TH1F("hT124LLLL", "Tmax124 LLL Dir.L", 2000, -1000., 1000.);
      hT124LRLL = new TH1F("hT124LRLL", "Tmax124 LRL Dir.L", 2000, -1000., 1000.);
      hT124LRLR = new TH1F("hT124LRLR", "Tmax124 LRL Dir.R", 2000, -1000., 1000.);
      hT124Bad = new TH1F("hT124Bad", "Bad Tmax124", 10, -5., 5.);
      hT134LLR1gt4 = new TH1F("hT134LLR1gt4", "Tmax134 LLR x1>x4", 2000, -1000., 1000.);
      hT134LLR1lt4 = new TH1F("hT134LLR1lt4", "Tmax134 LLR x1<x4", 2000, -1000., 1000.);
      hT134LRR = new TH1F("hT134LRR", "Tmax134", 2000, -1000., 1000.);
      hT134LRLR = new TH1F("hT134LRLR", "Tmax134 LRL Dir.R", 2000, -1000., 1000.);
      hT134LRLL = new TH1F("hT134LRLL", "Tmax134 LRL Dir.L", 2000, -1000., 1000.);
      hT134LLLL = new TH1F("hT134LLLL", "Tmax134 LLL Dir.L", 2000, -1000., 1000.);
      hT134LLLR = new TH1F("hT134LLLR", "Tmax134 LLL Dir.R", 2000, -1000., 1000.);
      hT134Bad = new TH1F("hT134Bad", "Bad Tmax134", 10, -5., 5.);
      hT234LRL = new TH1F("hT234LRL", "Tmax234 LRL", 2000, -1000., 1000.);
      hT234LLR = new TH1F("hT234LLR", "Tmax234 LLR", 2000, -1000., 1000.);
      hT234LRR = new TH1F("hT234LRR", "Tmax234 LRR", 2000, -1000., 1000.);
      hT234Bad = new TH1F("hT234Bad", "Bad Tmax234", 10, -5., 5.);
    }
  };
}  // namespace dtcalibration
#endif
