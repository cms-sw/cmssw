#ifndef RecoLocalTracker_ClusterRefinerTagMCmerged_h
#define RecoLocalTracker_ClusterRefinerTagMCmerged_h

//
// Use MC truth to identify merged clusters, i.e., those associated with more than one
// (in-time) SimTrack.
//
// Author:  Bill Ford (wtford)  6 March 2015
//

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

#include <memory>

class ClusterRefinerTagMCmerged : public edm::stream::EDProducer<>  {

public:

  explicit ClusterRefinerTagMCmerged(const edm::ParameterSet& conf);
  virtual void produce(edm::Event&, const edm::EventSetup&);

private:

  template<class T> bool findInput(const edm::EDGetTokenT<T>&, edm::Handle<T>&, const edm::Event&);
  virtual void refineCluster(const edm::Handle< edmNew::DetSetVector<SiStripCluster> >& input,
			     std::auto_ptr< edmNew::DetSetVector<SiStripCluster> >& output);

  const edm::InputTag inputTag;
  typedef edm::EDGetTokenT< edmNew::DetSetVector<SiStripCluster> > token_t;
  token_t inputToken;
  edm::ParameterSet confClusterRefiner_;
  bool useAssociateHit_; 

  TrackerHitAssociator::Config trackerHitAssociatorConfig_;
  std::unique_ptr<TrackerHitAssociator> associator_;

};

#endif
