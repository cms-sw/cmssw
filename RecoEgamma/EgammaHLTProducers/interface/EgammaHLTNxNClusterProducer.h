#ifndef RecoEcal_EgammaClusterProducers_EgammaHLTNxNClusterProducer_h_
#define RecoEcal_EgammaClusterProducers_EgammaHLTNxNClusterProducer_h_

/**

Description: simple NxN ( 3x3 etc) clustering ,( for low energy photon reconstrution, currently used for pi0/eta HLT path) 

 Implementation:
     <Notes on implementation>
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "DataFormats/CaloRecHit/interface/CaloID.h"

#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

#include <vector>
#include <memory>
#include <time.h>


namespace edm {
  class ConfigurationDescriptions;
}

// Less than operator for sorting EcalRecHits according to energy.
class ecalRecHitSort : public std::binary_function<EcalRecHit, EcalRecHit, bool> 
{
 public:
  bool operator()(EcalRecHit x, EcalRecHit y) 
  { 
    return (x.energy() > y.energy()); 
  }
};


class EgammaHLTNxNClusterProducer : public edm::stream::EDProducer<> {
 public:

  EgammaHLTNxNClusterProducer(const edm::ParameterSet& ps);
  ~EgammaHLTNxNClusterProducer();
  
  void produce(edm::Event&, const edm::EventSetup&) override ;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
  
  void makeNxNClusters(edm::Event &evt, const edm::EventSetup &es,const EcalRecHitCollection *hits, const reco::CaloID::Detectors detector); 
  
  bool checkStatusOfEcalRecHit(const EcalChannelStatus &channelStatus, const EcalRecHit &rh);
        
  //std::map<std::string,double> providedParameters;
      
  const bool doBarrel_;
  const bool doEndcaps_;  
  const edm::EDGetTokenT<EcalRecHitCollection> barrelHitProducer_;
  const edm::EDGetTokenT<EcalRecHitCollection> endcapHitProducer_;
  const int clusEtaSize_ ;
  const int clusPhiSize_;
  const std::string barrelClusterCollection_;
  const std::string endcapClusterCollection_;
  const double clusSeedThr_;
  const double clusSeedThrEndCap_;

  const bool useRecoFlag_; 
  const int flagLevelRecHitsToUse_; 
  const bool useDBStatus_; 
  const int statusLevelRecHitsToUse_;

  const int maxNumberofSeeds_ ; 
  const int maxNumberofClusters_; 

  const int debug_; 

  PositionCalc posCalculator_; // position calculation algorithm
};
#endif
