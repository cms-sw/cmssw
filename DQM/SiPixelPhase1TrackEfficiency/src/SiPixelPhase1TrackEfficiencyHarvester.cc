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
  // We collect _all_ (all specs/all custom calls) histos from missing/valid in our table
  histo[VALID     ].setCustomHandler([&] (SummationStep const& s, HistogramManager::Table& t,
                                          DQMStore::IBooker&, DQMStore::IGetter&) {
    valid  [s.arg].insert(t.begin(), t.end());
  });
  histo[MISSING   ].setCustomHandler([&] (SummationStep const& s, HistogramManager::Table& t,
                                          DQMStore::IBooker&, DQMStore::IGetter&) {
    missing[s.arg].insert(t.begin(), t.end());
  });

  // ... and then take those that we need to fill the EFFICIENCY
  // note: we don't need the iBooker here, since the eff. histograms are booked with a HistogramManager
  histo[EFFICIENCY].setCustomHandler([&] (SummationStep const& s, HistogramManager::Table& t,
                                          DQMStore::IBooker&, DQMStore::IGetter&) {
    doHarvesting(s, t);
  });
}

void SiPixelPhase1TrackEfficiencyHarvester::doHarvesting(SummationStep const& s, HistogramManager::Table& efficiency) {
  for (auto const& e : efficiency) {
    GeometryInterface::Values const& values = e.first;
    auto missing_it = missing[s.arg].find(values);
    auto valid_it   = valid  [s.arg].find(values);
    if (missing_it == missing[s.arg].end() || valid_it == valid[s.arg].end()) {
      edm::LogError("SiPixelPhase1TrackEfficiencyHarvester") << "Want to do Efficiencies but 'valid' or 'missing' counts are missing.";
      continue;
    }
    auto num = missing_it->second.th1;
    auto denom = valid_it->second.th1;
    assert(num);
    assert(denom);
    assert(num->GetDimension() == denom->GetDimension());

    auto& out = efficiency[values];
    assert(out.th1);
    assert(out.th1->GetDimension() == num->GetDimension());

    if (num->GetDimension() == 1) {
      auto xbins = num->GetXaxis()->GetNbins();
      assert(denom->GetXaxis()->GetNbins() == xbins || out.th1->GetXaxis()->GetNbins());

      for (int x = 1; x <= xbins; x++) {
        auto sum = num->GetBinContent(x) + denom->GetBinContent(x);
        if (sum == 0.0) continue; // avoid div by zero
        out.th1->SetBinContent(x, 1 - (num->GetBinContent(x) / sum));
      }
 
    } else /* 2D */ {
      auto xbins = num->GetXaxis()->GetNbins();
      auto ybins = num->GetYaxis()->GetNbins();
      assert(denom->GetXaxis()->GetNbins() == xbins || out.th1->GetXaxis()->GetNbins());
      assert(denom->GetYaxis()->GetNbins() == ybins || out.th1->GetYaxis()->GetNbins());

      for (int y = 1; y <= ybins; y++) {
        for (int x = 1; x <= xbins; x++) {
          auto sum = num->GetBinContent(x,y) + denom->GetBinContent(x,y);
          if (sum == 0.0) continue; // avoid div by zero
          out.th1->SetBinContent(x, y, 1 - (num->GetBinContent(x,y) / sum));
        }
      }
    }
  }
}

DEFINE_FWK_MODULE(SiPixelPhase1TrackEfficiencyHarvester);

