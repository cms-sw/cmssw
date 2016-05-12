// -*- C++ -*-
//
// Package:     SiPixelPhase1TrackEfficiencyHarvester
// Class:       SiPixelPhase1TrackEfficiencyHarvester
//

// Original Author: Marcel Schneider

#include "DQM/SiPixelPhase1TrackEfficiency/interface/SiPixelPhase1TrackEfficiency.h"
#include "FWCore/Framework/interface/MakerMacros.h"

SiPixelPhase1TrackEfficiencyHarvester::SiPixelPhase1TrackEfficiencyHarvester(const edm::ParameterSet& iConfig) :
  SiPixelPhase1Harvester(iConfig) 
{
  histo[VALID     ].setCustomHandler([&] (SummationStep&, HistogramManager::Table & t) {
    valid.insert(t.begin(), t.end());
  });
  histo[MISSING   ].setCustomHandler([&] (SummationStep&, HistogramManager::Table & t) {
    missing.insert(t.begin(), t.end());
  });

  histo[EFFICIENCY].setCustomHandler([&] (SummationStep&, HistogramManager::Table & t) {
    doHarvesting();
    t.swap(efficiency);
    efficiency.clear();
  });
}

void SiPixelPhase1TrackEfficiencyHarvester::doHarvesting() {
  for (auto const& e : valid) {
    GeometryInterface::Values const& values = e.first;
    auto denom = e.second.th1;
    auto missing_it = missing.find(values);
    if (missing_it == missing.end()) {
      edm::LogError("SiPixelPhase1TrackEfficiencyHarvester") << "Got 'valid' counts, but 'missing' counts are missing.\n";
      continue;
    }
    auto num = missing_it->second.th1;
    assert(num);
    assert(denom);
    assert(num->GetDimension() == denom->GetDimension());

    auto& out = efficiency[values];
    if (num->GetDimension() == 1) {
      auto title = (histo[EFFICIENCY].title + ";" + num->GetXaxis()->GetTitle()).c_str();
      auto xbins = num->GetXaxis()->GetNbins();
      assert(denom->GetXaxis()->GetNbins() == xbins);

      out.th1 = (TH1*) new TH1F(histo[EFFICIENCY].name.c_str(), title, xbins, 0.5, xbins + 0.5);
      for (int x = 1; x <= xbins; x++) {
        out.th1->SetBinContent(x, 1 - (num->GetBinContent(x) / (num->GetBinContent(x) + denom->GetBinContent(x))));
      }
 
    } else /* 2D */ {
      auto title = (histo[EFFICIENCY].title + ";" + num->GetXaxis()->GetTitle() + ";" + num->GetYaxis()->GetTitle()).c_str();
      auto xbins = num->GetXaxis()->GetNbins();
      auto ybins = num->GetYaxis()->GetNbins();
      assert(denom->GetXaxis()->GetNbins() == xbins);
      assert(denom->GetYaxis()->GetNbins() == ybins);

      out.th1 = (TH1*) new TH2F(histo[EFFICIENCY].name.c_str(), title, xbins, 0.5, xbins + 0.5,
                                                                       ybins, 0.5, ybins + 0.5);
      for (int y = 1; y <= ybins; y++) {
        for (int x = 1; x <= xbins; x++) {
          out.th1->SetBinContent(x, y, 1 - (num->GetBinContent(x,y) / (num->GetBinContent(x,y) + denom->GetBinContent(x,y))));
        }
      }
    }
  }
}

DEFINE_FWK_MODULE(SiPixelPhase1TrackEfficiencyHarvester);

