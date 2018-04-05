#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/METReco/interface/BeamHaloSummary.h"
#include "DataFormats/METReco/interface/HcalHaloData.h"

class HcalStripHaloFilter : public edm::global::EDFilter<> {

  public:

    explicit HcalStripHaloFilter(const edm::ParameterSet & iConfig);
    ~HcalStripHaloFilter() override {}

  private:

    bool filter(edm::StreamID iID, edm::Event & iEvent, const edm::EventSetup & iSetup) const override;

    const bool taggingMode_;
    const int maxWeightedStripLength_;
    const double maxEnergyRatio_;
    const double minHadEt_;
    edm::EDGetTokenT<reco::BeamHaloSummary> beamHaloSummaryToken_;
};

HcalStripHaloFilter::HcalStripHaloFilter(const edm::ParameterSet & iConfig)
  : taggingMode_     (iConfig.getParameter<bool> ("taggingMode"))
  , maxWeightedStripLength_   (iConfig.getParameter<int>("maxWeightedStripLength"))
  , maxEnergyRatio_   (iConfig.getParameter<double>("maxEnergyRatio"))
  , minHadEt_   (iConfig.getParameter<double>("minHadEt"))
  , beamHaloSummaryToken_(consumes<reco::BeamHaloSummary>(edm::InputTag("BeamHaloSummary")))
{
  produces<bool>();
}

bool HcalStripHaloFilter::filter(edm::StreamID iID, edm::Event & iEvent, const edm::EventSetup & iSetup) const {

  edm::Handle<reco::BeamHaloSummary> beamHaloSummary;
  iEvent.getByToken(beamHaloSummaryToken_ , beamHaloSummary);

  bool pass = true;
  auto const& problematicStrips = beamHaloSummary->getProblematicStrips();
  for (unsigned int iStrip = 0; iStrip < problematicStrips.size(); iStrip++) {
    int numContiguousCells = 0;
    auto const& problematicStrip = problematicStrips[iStrip];
    for (unsigned int iTower = 0; iTower < problematicStrip.cellTowerIds.size(); iTower++) {
      numContiguousCells += (int)problematicStrip.cellTowerIds[iTower].first;
    }
    if(   numContiguousCells > maxWeightedStripLength_ 
       && problematicStrip.energyRatio < maxEnergyRatio_ 
       && problematicStrip.hadEt > minHadEt_ ) {
      pass = false;
      break;
    }
  }

  iEvent.put(std::make_unique<bool>(pass));

  return taggingMode_ || pass;  // return false if it is a beamhalo event
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HcalStripHaloFilter);
