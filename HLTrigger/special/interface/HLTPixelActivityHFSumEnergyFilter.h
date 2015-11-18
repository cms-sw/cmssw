#ifndef _HLTPixelActivityHFSumEnergyFilter_H
#define _HLTPixelActivityHFSumEnergyFilter_H

// system include files
#include <memory>

// user include files
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

//
// class declaration
//

class HLTPixelActivityHFSumEnergyFilter : public HLTFilter {
public:
  explicit HLTPixelActivityHFSumEnergyFilter(const edm::ParameterSet&);
  ~HLTPixelActivityHFSumEnergyFilter();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

  edm::InputTag inputTag_;          // input tag identifying product containing pixel digis
  //edm::EDGetTokenT<edmNew::DetSetVector<PixelDigi> > inputToken_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > inputToken_;
  edm::EDGetTokenT<HFRecHitCollection> HFHitsToken_;
  edm::InputTag HFHits_;
  double eCut_HF_;
  double eMin_HF_;
  double offset_;
  double slope_;

};

#endif
