#ifndef _HLTPixelActivityHFSumEnergyFilter_H
#define _HLTPixelActivityHFSumEnergyFilter_H

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/stream/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

//
// class declaration
//

class HLTPixelActivityHFSumEnergyFilter : public edm::stream::EDFilter<> {
public:
  explicit HLTPixelActivityHFSumEnergyFilter(const edm::ParameterSet&);
  ~HLTPixelActivityHFSumEnergyFilter() override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  bool filter(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::InputTag inputTag_;   // input tag identifying product containing pixel clusters
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > inputToken_;
  edm::EDGetTokenT<HFRecHitCollection> HFHitsToken_;
  edm::InputTag HFHits_;
  double eCut_HF_;
  double eMin_HF_;
  double offset_;
  double slope_;
};

#endif
