#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/METReco/interface/BeamHaloSummary.h"

class GlobalTightHalo2016Filter : public edm::global::EDFilter<> {

  public:

    explicit GlobalTightHalo2016Filter(const edm::ParameterSet & iConfig);
    ~GlobalTightHalo2016Filter() {}

  private:

  virtual bool filter(edm::StreamID iID, edm::Event & iEvent, const edm::EventSetup & iSetup) const override;

    const bool taggingMode_;
    edm::EDGetTokenT<reco::BeamHaloSummary> beamHaloSummaryToken_;
};

GlobalTightHalo2016Filter::GlobalTightHalo2016Filter(const edm::ParameterSet & iConfig)
  : taggingMode_     (iConfig.getParameter<bool> ("taggingMode"))
  , beamHaloSummaryToken_(consumes<reco::BeamHaloSummary>(edm::InputTag("BeamHaloSummary")))
{

  produces<bool>();
}

bool GlobalTightHalo2016Filter::filter(edm::StreamID iID, edm::Event & iEvent, const edm::EventSetup & iSetup) const {

  edm::Handle<reco::BeamHaloSummary> beamHaloSummary;
  iEvent.getByToken(beamHaloSummaryToken_ , beamHaloSummary);

  const bool pass = !beamHaloSummary->GlobalTightHaloId2016();

  iEvent.put( std::auto_ptr<bool>(new bool(pass)) );

  return taggingMode_ || pass;  // return false if it is a beamhalo event
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GlobalTightHalo2016Filter);
