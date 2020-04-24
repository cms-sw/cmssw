#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/METReco/interface/BeamHaloSummary.h"

class GlobalSuperTightHalo2016Filter : public edm::global::EDFilter<> {

  public:

    explicit GlobalSuperTightHalo2016Filter(const edm::ParameterSet & iConfig);
    ~GlobalSuperTightHalo2016Filter() override {}

  private:

  bool filter(edm::StreamID iID, edm::Event & iEvent, const edm::EventSetup & iSetup) const override;

    const bool taggingMode_;
    edm::EDGetTokenT<reco::BeamHaloSummary> beamHaloSummaryToken_;
};

GlobalSuperTightHalo2016Filter::GlobalSuperTightHalo2016Filter(const edm::ParameterSet & iConfig)
  : taggingMode_     (iConfig.getParameter<bool> ("taggingMode"))
  , beamHaloSummaryToken_(consumes<reco::BeamHaloSummary>(edm::InputTag("BeamHaloSummary")))
{

  produces<bool>();
}

bool GlobalSuperTightHalo2016Filter::filter(edm::StreamID iID, edm::Event & iEvent, const edm::EventSetup & iSetup) const {

  edm::Handle<reco::BeamHaloSummary> beamHaloSummary;
  iEvent.getByToken(beamHaloSummaryToken_ , beamHaloSummary);

  const bool pass = !beamHaloSummary->GlobalSuperTightHaloId2016();

  iEvent.put(std::make_unique<bool>(pass));

  return taggingMode_ || pass;  // return false if it is a beamhalo event
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GlobalSuperTightHalo2016Filter);
