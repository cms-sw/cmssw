#ifndef PhotonIDAlgo_H
#define PhotonIDAlgo_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonID.h"
#include <string>

class PhotonIDAlgo {

public:

  PhotonIDAlgo(){};

  virtual ~PhotonIDAlgo(){};

  void baseSetup(const edm::ParameterSet& conf);
  void classify(const reco::Photon* photon, 
		bool &isEBPho,
		bool &isEEPho,
		bool &isEBGap,
		bool &isEEGap,
		bool &isEBEEGap);
  void calculateTrackIso(const reco::Photon* photon,
			 const edm::Event &e,
			 double &trkCone,
			 int &ntrkCone,
			 double pTThresh=0,
			 double RCone=.4,
			 double RinnerCone=.1);
  double calculateBasicClusterIso(const reco::Photon* photon,
				  const edm::Event& iEvent,
				  double RCone=0.4,
				  double RConeInner=0,
				  double etMin=0);
  double calculateEcalRecHitIso(const reco::Photon* photon,
				const edm::Event& iEvent,
				const edm::EventSetup& iSetup,
				double RCone,
				double RConeInner,
				double etMin);
  double calculateHcalRecHitIso(const reco::Photon* photon,
				const edm::Event& iEvent,
				const edm::EventSetup& iSetup,
				double RCone,
				double RConeInner,
				double etMin);
  bool isAlsoElectron(const reco::Photon* photon,
		      const edm::Event& e);
  
  
 protected:


  std::string endcapSuperClusterProducer_;      
  std::string endcapsuperclusterCollection_;
  std::string barrelislandsuperclusterCollection_;
  std::string barrelislandsuperclusterProducer_;
  std::string barrelbasicclusterCollection_;
  std::string barrelbasicclusterProducer_;
  std::string endcapbasicclusterCollection_;
  std::string endcapbasicclusterProducer_;
  std::string barrelecalCollection_;
  std::string barrelecalProducer_;
  std::string endcapecalCollection_;
  std::string endcapecalProducer_;
  std::string hcalCollection_;
  std::string hcalProducer_;

  

  edm::InputTag trackInputTag_;
  edm::InputTag gsfRecoInputTag_;


  };

#endif // PhotonIDAlgo_H
