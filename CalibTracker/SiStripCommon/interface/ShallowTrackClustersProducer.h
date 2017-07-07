#ifndef SHALLOW_TRACKCLUSTERS_PRODUCER
#define SHALLOW_TRACKCLUSTERS_PRODUCER

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

class ShallowTrackClustersProducer : public edm::EDProducer {
public:
  explicit ShallowTrackClustersProducer(const edm::ParameterSet&);
private:
	const edm::EDGetTokenT<edm::View<reco::Track> > tracks_token_;
	const edm::EDGetTokenT<TrajTrackAssociationCollection> association_token_;
	const edm::EDGetTokenT< edmNew::DetSetVector<SiStripCluster> > clusters_token_;
  std::string Suffix;
  std::string Prefix;

  void produce( edm::Event &, const edm::EventSetup & ) override;
};
#endif
