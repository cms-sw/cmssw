// Harvester with a custom step to reset histograms.

#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1Base.h"
#include "FWCore/Framework/interface/MakerMacros.h"

class SiPixelPhase1OnlineHarvester : public SiPixelPhase1Harvester {

  public:
  SiPixelPhase1OnlineHarvester(const edm::ParameterSet& iConfig) 
   : SiPixelPhase1Harvester(iConfig)  {

    for (auto& h : histo) {
      h.setCustomHandler([&h] (SummationStep& s, HistogramManager::Table & t) {
        if (!h.lumisection) return; // not online
        uint32_t ls = h.lumisection->id().luminosityBlock();
        // TODO: keep in sync with Geometry interface
        uint32_t block = (ls / 10) % 3;
        uint32_t next_block = ((ls / 10)+1) % 3;
        if (block == next_block) return;

        for (auto& e : t) {
          TH1* th1 = e.second.th1;
          if (std::string(th1->GetYaxis()->GetTitle()).find("OnlineBlock") != std::string::npos) {
            for (int i = 1; i <= th1->GetNbinsX(); i++) {
              th1->SetBinContent(i, next_block+1, 0);
            }
          } else {
            // TODO: do sth.like exponential decay here.
          }
        }
      });
    }
  }

};

DEFINE_FWK_MODULE(SiPixelPhase1OnlineHarvester);

