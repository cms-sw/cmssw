#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

class QualityFilter : public edm::EDProducer {
 public:
  explicit QualityFilter(const edm::ParameterSet&);
  ~QualityFilter();
  
 private:
  virtual void beginRun(edm::Run & run, const edm::EventSetup&) ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  // ----------member data ---------------------------
 private:
  
  edm::InputTag tkTag; 
  reco::TrackBase::TrackQuality trackQuality_;
  
};
