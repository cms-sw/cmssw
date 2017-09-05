/**********************************************************************
 *
 * Author: F.Ferro - INFN Genova
 *
 **********************************************************************/
#ifndef RecoCTPPS_PixelLocal_CTPPSPixelRecHitProducer_H
#define RecoCTPPS_PixelLocal_CTPPSPixelRecHitProducer_H

#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSet.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
 
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/CTPPSReco/interface/CTPPSPixelCluster.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelRecHit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"
#include "RecoCTPPS/PixelLocal/interface/RPixClusterToHit.h" 


class CTPPSPixelRecHitProducer : public edm::stream::EDProducer<>
{
public:
  explicit CTPPSPixelRecHitProducer(const edm::ParameterSet& param);
  
  ~CTPPSPixelRecHitProducer() override;
  
  void produce(edm::Event&, const edm::EventSetup&) override;
  
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  edm::ParameterSet param_;
  int verbosity_;
  
  edm::InputTag src_;
  edm::EDGetTokenT<edm::DetSetVector<CTPPSPixelCluster>> tokenCTPPSPixelCluster_;
  
  RPixClusterToHit cluster2hit_;
  
  void run(const edm::DetSetVector<CTPPSPixelCluster> &input, edm::DetSetVector<CTPPSPixelRecHit> &output);
};

#endif
