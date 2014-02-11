#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/METReco/interface/BeamHaloSummary.h"

class CSCTightHaloFilter : public edm::EDFilter {

  public:

    explicit CSCTightHaloFilter(const edm::ParameterSet & iConfig);
    ~CSCTightHaloFilter() {}

  private:

    virtual bool filter(edm::Event & iEvent, const edm::EventSetup & iSetup);

    const bool taggingMode_;
};

CSCTightHaloFilter::CSCTightHaloFilter(const edm::ParameterSet & iConfig)
  : taggingMode_     (iConfig.getParameter<bool> ("taggingMode"))
{

  produces<bool>();
}

bool CSCTightHaloFilter::filter(edm::Event & iEvent, const edm::EventSetup & iSetup) {

  edm::Handle<reco::BeamHaloSummary> beamHaloSummary;
  iEvent.getByLabel("BeamHaloSummary" , beamHaloSummary);

  const bool pass = !beamHaloSummary->CSCTightHaloId();

  iEvent.put( std::auto_ptr<bool>(new bool(pass)) );

  return taggingMode_ || pass;  // return false if it is a beamhalo event
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CSCTightHaloFilter);
