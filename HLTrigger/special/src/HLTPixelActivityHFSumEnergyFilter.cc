#include "HLTrigger/special/interface/HLTPixelActivityHFSumEnergyFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//

HLTPixelActivityHFSumEnergyFilter::HLTPixelActivityHFSumEnergyFilter(const edm::ParameterSet& config) :
  inputTag_     (config.getParameter<edm::InputTag>("inputTag")),
  HFHits_       (config.getParameter<edm::InputTag>("HFHitCollection")),  
  eCut_HF_      (config.getParameter<double>("eCut_HF")),
  eMin_HF_      (config.getParameter<double>("eMin_HF")),
  offset_       (config.getParameter<double>("offset")),
  slope_        (config.getParameter<double>("slope"))
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
  desc.add<edm::InputTag>("inputTag",edm::InputTag("hltSiPixelClusters"));
  desc.add<edm::InputTag>("HFHitCollection",edm::InputTag("hltHfreco"));
  desc.add<double>("eCut_HF",0);
  desc.add<double>("eMin_HF",10000.);
  desc.add<double>("offset",-1000.);
  desc.add<double>("slope",0.5);
  descriptions.add("hltPixelActivityHFSumEnergyFilter",desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool HLTPixelActivityHFSumEnergyFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // get hold of products from Event
  edm::Handle<edmNew::DetSetVector<SiPixelCluster> > clusterColl;
  iEvent.getByToken(inputToken_, clusterColl);

  unsigned int clusterSize = clusterColl->dataSize();
  LogDebug("") << "Number of clusters: " << clusterSize;

  edm::Handle<HFRecHitCollection> HFRecHitsH;
  iEvent.getByToken(HFHitsToken_,HFRecHitsH);

  double sumE = 0.;

  for (HFRecHitCollection::const_iterator it=HFRecHitsH->begin(); it!=HFRecHitsH->end(); it++) {
    if (it->energy()>eCut_HF_) {
      sumE += it->energy();
    }
  }

  bool accept = false;

  double thres = offset_ + slope_ * sumE;
  double diff = clusterSize - thres;    //diff = clustersize - (correlation line + offset)
  if(sumE>eMin_HF_ && diff < 0.) accept = true;
  
  // return with final filter decision
  return accept;
}
