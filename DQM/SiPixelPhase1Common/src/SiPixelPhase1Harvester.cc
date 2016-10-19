// This is a plugin implementation, but it is in src/ to make it possible to 
// derive from it in other packages. In plugins/ there is a dummy that declares
// the plugin.
#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1Base.h"
#include "FWCore/Framework/interface/MakerMacros.h"

void SiPixelPhase1Harvester::dqmEndLuminosityBlock(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter, edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& eSetup) {
  for (HistogramManager& histoman : histo)
    histoman.executePerLumiHarvesting(iBooker, iGetter, eSetup);
};
void SiPixelPhase1Harvester::dqmEndJob(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter) {
  for (HistogramManager& histoman : histo)
    histoman.executeHarvesting(iBooker, iGetter);
};
