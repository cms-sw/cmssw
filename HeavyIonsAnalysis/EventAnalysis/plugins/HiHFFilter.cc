// system include files
#include <memory>
#include <vector>
#include <map>
#include <set>

// user include files
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/HeavyIonEvent/interface/HFFilterInfo.h"  //this line is needed to access the HF Filters

class HiHFFilter : public edm::one::EDFilter<> {
public:
  explicit HiHFFilter(const edm::ParameterSet&);
  ~HiHFFilter() override;

private:
  bool filter(edm::Event&, const edm::EventSetup&) override;

  edm::EDGetTokenT<reco::HFFilterInfo> HFfilters_;
  bool applyfilter_;
  int threshold_;
  int minnumtowers_;
  int numMinHFTowers;
};

using namespace edm;
using namespace std;

HiHFFilter::HiHFFilter(const edm::ParameterSet& iConfig) {
  HFfilters_ = consumes<reco::HFFilterInfo>(iConfig.getParameter<edm::InputTag>("HFfilters"));
  //  applyfilter_ = iConfig.getParameter<bool>("applyfilter");
  threshold_ = iConfig.getParameter<int>("threshold");
  minnumtowers_ = iConfig.getParameter<int>("minnumtowers");
}

HiHFFilter::~HiHFFilter() {}

bool HiHFFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  bool accepted = false;

  numMinHFTowers = 0;

  edm::Handle<reco::HFFilterInfo> HFfilter;
  iEvent.getByToken(HFfilters_, HFfilter);

  if (threshold_ == 2) {
    numMinHFTowers = HFfilter->numMinHFTowers2;
  }
  if (threshold_ == 3) {
    numMinHFTowers = HFfilter->numMinHFTowers3;
  }
  if (threshold_ == 4) {
    numMinHFTowers = HFfilter->numMinHFTowers4;
  }
  if (threshold_ == 5) {
    numMinHFTowers = HFfilter->numMinHFTowers5;
  }

  if (numMinHFTowers >= minnumtowers_)
    accepted = true;

  return accepted;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HiHFFilter);
