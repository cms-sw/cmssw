#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/METReco/interface/BeamHaloSummary.h"

class CSCTightHalo2015Filter : public edm::global::EDFilter<> {

  public:

    explicit CSCTightHalo2015Filter(const edm::ParameterSet & iConfig);
    ~CSCTightHalo2015Filter() override {}

  private:

  bool filter(edm::StreamID iID, edm::Event & iEvent, const edm::EventSetup & iSetup) const override;

    const bool taggingMode_;
    edm::EDGetTokenT<reco::BeamHaloSummary> beamHaloSummaryToken_;
};

CSCTightHalo2015Filter::CSCTightHalo2015Filter(const edm::ParameterSet & iConfig)
  : taggingMode_     (iConfig.getParameter<bool> ("taggingMode"))
  , beamHaloSummaryToken_(consumes<reco::BeamHaloSummary>(edm::InputTag("BeamHaloSummary")))
{

  produces<bool>();
}

bool CSCTightHalo2015Filter::filter(edm::StreamID iID, edm::Event & iEvent, const edm::EventSetup & iSetup) const {

  edm::Handle<reco::BeamHaloSummary> beamHaloSummary;
  iEvent.getByToken(beamHaloSummaryToken_ , beamHaloSummary);

  const bool pass = !beamHaloSummary->CSCTightHaloId2015();

  iEvent.put(std::make_unique<bool>(pass));

  return taggingMode_ || pass;  // return false if it is a beamhalo event
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CSCTightHalo2015Filter);
