#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

class QualityFilter : public edm::EDProducer {
 public:
  explicit QualityFilter(const edm::ParameterSet&);
  ~QualityFilter();
  
 private:
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() ;
  
  // ----------member data ---------------------------
 private:
  
  edm::EDGetTokenT<std::vector<Trajectory> >       trajTag; 
  edm::EDGetTokenT<TrajTrackAssociationCollection> tassTag; 
  reco::TrackBase::TrackQuality trackQuality_;
  bool copyExtras_;  
};
