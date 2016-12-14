// Harvester with a custom step to reset histograms.

#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1Base.h"
#include "FWCore/Framework/interface/MakerMacros.h"

class SiPixelPhase1OnlineHarvester : public SiPixelPhase1Harvester {
  // Note: for the current setup of onlineblocks (for overlaid curves,
  // e.g. useful for detector commisioning), this plugin is NOT needed.

  public:
  SiPixelPhase1OnlineHarvester(const edm::ParameterSet& iConfig) 
   : SiPixelPhase1Harvester(iConfig)  {

    int onlineblock = iConfig.getParameter<edm::ParameterSet>("geometry").getParameter<int>("onlineblock");
    int n_onlineblocks = iConfig.getParameter<edm::ParameterSet>("geometry").getParameter<int>("n_onlineblocks");

    for (auto& h : histo) {
      h.setCustomHandler([&h, onlineblock, n_onlineblocks] (SummationStep const& s, HistogramManager::Table & t,
                                                            DQMStore::IBooker&, DQMStore::IGetter&) {
        if (!h.lumisection) return; // not online
        uint32_t ls = h.lumisection->id().luminosityBlock();

        // TODO: this is hard to get right if we don't see all LS.
        uint32_t block = (ls / onlineblock) % n_onlineblocks;
        uint32_t next_block = ((ls / onlineblock)+1) % n_onlineblocks;
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

