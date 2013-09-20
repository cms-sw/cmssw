#ifndef RecoEcal_EgammaClusterProducers_EgammaHLTNxNClusterProducer_h_
#define RecoEcal_EgammaClusterProducers_EgammaHLTNxNClusterProducer_h_

/**

Description: simple NxN ( 3x3 etc) clustering ,( for low energy photon reconstrution, currently used for pi0/eta HLT path) 

 Implementation:
     <Notes on implementation>
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
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


// Less than operator for sorting EcalRecHits according to energy.
class ecalRecHitSort : public std::binary_function<EcalRecHit, EcalRecHit, bool> 
{
 public:
  bool operator()(EcalRecHit x, EcalRecHit y) 
  { 
    return (x.energy() > y.energy()); 
  }
};


class EgammaHLTNxNClusterProducer : public edm::EDProducer {
 public:

  EgammaHLTNxNClusterProducer(const edm::ParameterSet& ps);
  ~EgammaHLTNxNClusterProducer();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);
      
 private:
  
  void makeNxNClusters(edm::Event &evt, const edm::EventSetup &es,const EcalRecHitCollection *hits, const reco::CaloID::Detectors detector); 
  
  bool checkStatusOfEcalRecHit(const EcalChannelStatus &channelStatus, const EcalRecHit &rh);
  
  
  std::string barrelClusterCollection_;
  std::string endcapClusterCollection_;
  
  std::string barrelHits_;
  std::string endcapHits_;
        
  PositionCalc posCalculator_; // position calculation algorithm
  std::map<std::string,double> providedParameters;
  
  edm::EDGetTokenT<EcalRecHitCollection> barrelHitProducer_;
  edm::EDGetTokenT<EcalRecHitCollection> endcapHitProducer_;
      
  double clusSeedThr_;
  double clusSeedThrEndCap_;
      
  bool doBarrel_;
  bool doEndcaps_;
  
  bool useRecoFlag_; 
  bool useDBStatus_; 
  int flagLevelRecHitsToUse_; 
  int statusLevelRecHitsToUse_;
  
  int clusEtaSize_ ;
  int clusPhiSize_;
        
  int debug_; 
  
  int maxNumberofSeeds_ ; 
  int maxNumberofClusters_; 
};
#endif
