// -*- C++ -*-
//
// Package:     SiPixelPhase1DigisHarvester
// Class:       SiPixelPhase1DigisHarvester
//

// Original Author: Marcel Schneider

#include "DQM/SiPixelPhase1Digis/interface/SiPixelPhase1Digis.h"
#include "FWCore/Framework/interface/MakerMacros.h"

SiPixelPhase1DigisHarvester::SiPixelPhase1DigisHarvester(const edm::ParameterSet& iConfig) :
  SiPixelPhase1Harvester(iConfig) 
{
  histo[NDIGIS_FED].setCustomHandler([&] (SummationStep const& s, HistogramManager::Table& t,
                                          DQMStore::IBooker& iBooker, DQMStore::IGetter&) {
    for (auto e : t) {
      TH1* th1 = e.second.th1;
      assert(th1->GetDimension() == 2);
      TH2D* th2 = dynamic_cast<TH2D*>(th1);
      assert(th2);
      iBooker.setCurrentFolder(e.second.me->getPathname());
      MonitorElement* out = iBooker.book2DD(e.second.me->getName() + "_norm", th2);

      for (int x = 1; x <= th1->GetNbinsX(); x++) {
        double sum = 0;
        int nonzero = 0;
        for (int y = 1; y <= th1->GetNbinsY(); y++) {
          double val = th1->GetBinContent(x, y);
          sum += val;
          if (val != 0.0) nonzero++;
        }

        if (nonzero == 0) continue;

        double avg = sum / nonzero;

        for (int y = 1; y <= th1->GetNbinsY(); y++) {
          out->setBinContent(x, y, th1->GetBinContent(x, y) / avg);
        }
      }
    }
  });
}

DEFINE_FWK_MODULE(SiPixelPhase1DigisHarvester);
