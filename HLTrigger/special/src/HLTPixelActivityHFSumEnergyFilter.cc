#include "HLTrigger/special/interface/HLTPixelActivityHFSumEnergyFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "FWCore/Utilities/interface/StreamID.h"

//
// constructors and destructor
//

HLTPixelActivityHFSumEnergyFilter::HLTPixelActivityHFSumEnergyFilter(const edm::ParameterSet& config) : HLTFilter(config),                                                                            inputTag_     (config.getParameter<edm::InputTag>("inputTag")),
  HFHits_       (config.getParameter<edm::InputTag>("HFHitCollection")),  
  eCut_HF_      (config.getParameter<double>("eCut_HF")),
  eMin_HF_      (config.getParameter<double>("eMin_HF")),
  offset_       (config.getParameter<double>("offset")),
  slope_        (config.getParameter<double>("slope")),
  maxDiff_      (config.getParameter<double>("maxDiff"))
{
  inputToken_ = consumes<edmNew::DetSetVector<SiPixelCluster> >(inputTag_);
  HFHitsToken_ = consumes<HFRecHitCollection>(HFHits_); 
  
  LogDebug("") << "Using the " << inputTag_ << " input collection";
}

HLTPixelActivityHFSumEnergyFilter::~HLTPixelActivityHFSumEnergyFilter()
{
}

void
HLTPixelActivityHFSumEnergyFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputTag",edm::InputTag("hltSiPixelClusters"));
  desc.add<edm::InputTag>("HFHitCollection",edm::InputTag("hltHfreco"));
  desc.add<double>("eCut_HF",0);
  desc.add<double>("eMin_HF",0);
  desc.add<double>("offset",0);
  desc.add<double>("slope",0);
  desc.add<double>("maxDiff",1e-5);
  descriptions.add("hltPixelActivityHFSumEnergyFilter",desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool HLTPixelActivityHFSumEnergyFilter::hltFilter(edm::Event& event, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{
  // All HLT filters must create and fill an HLT filter object,
  // recording any reconstructed physics objects satisfying (or not)
  // this HLT filter, and place it in the Event.

  // The filter object
  if (saveTags()) {
	filterproduct.addCollectionTag(inputTag_);
        filterproduct.addCollectionTag(HFHits_);
  }

  // get hold of products from Event
  edm::Handle<edmNew::DetSetVector<SiPixelCluster> > clusterColl;
  event.getByToken(inputToken_, clusterColl);

  unsigned int clusterSize = clusterColl->dataSize();
  LogDebug("") << "Number of clusters: " << clusterSize;

  edm::Handle<HFRecHitCollection> HFRecHitsH;
  event.getByToken(HFHitsToken_,HFRecHitsH);

  double sumE = 0.;

  for (HFRecHitCollection::const_iterator it=HFRecHitsH->begin(); it!=HFRecHitsH->end(); it++) {
    if (it->energy()>eCut_HF_) {
      sumE += it->energy();
    }
  }

  bool accept = kFALSE;

  double thres = offset_ + slope_ * sumE;
  double diff = clusterSize - thres;
  if(sumE>eMin_HF_ && diff < maxDiff_) accept = kTRUE;
  
  // return with final filter decision
  return accept;
}
