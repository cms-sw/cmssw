#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/METReco/interface/BeamHaloSummary.h"

class CSCTightHaloTrkMuUnvetoFilter : public edm::global::EDFilter<> {

  public:

    explicit CSCTightHaloTrkMuUnvetoFilter(const edm::ParameterSet & iConfig);
    ~CSCTightHaloTrkMuUnvetoFilter() {}

  private:

  virtual bool filter(edm::StreamID iID, edm::Event & iEvent, const edm::EventSetup & iSetup) const override;

    const bool taggingMode_;
    edm::EDGetTokenT<reco::BeamHaloSummary> beamHaloSummaryToken_;
};

CSCTightHaloTrkMuUnvetoFilter::CSCTightHaloTrkMuUnvetoFilter(const edm::ParameterSet & iConfig)
  : taggingMode_     (iConfig.getParameter<bool> ("taggingMode"))
  , beamHaloSummaryToken_(consumes<reco::BeamHaloSummary>(edm::InputTag("BeamHaloSummary")))
{

  produces<bool>();
}

bool CSCTightHaloTrkMuUnvetoFilter::filter(edm::StreamID iID, edm::Event & iEvent, const edm::EventSetup & iSetup) const {

  edm::Handle<reco::BeamHaloSummary> beamHaloSummary;
  iEvent.getByToken(beamHaloSummaryToken_ , beamHaloSummary);

  const bool pass = !beamHaloSummary->CSCTightHaloIdTrkMuUnveto();

  iEvent.put( std::auto_ptr<bool>(new bool(pass)) );

  return taggingMode_ || pass;  // return false if it is a beamhalo event
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CSCTightHaloTrkMuUnvetoFilter);
