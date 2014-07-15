#ifndef PhotonIsolationCalculator_H
#define PhotonIsolationCalculator_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"

#include <string>

#include "FWCore/Utilities/interface/GCC11Compatibility.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

class PhotonIsolationCalculator {

public:

  PhotonIsolationCalculator(){}

  ~PhotonIsolationCalculator(){}

  void setup(const edm::ParameterSet& conf,
	     std::vector<int> const & flagsEB_,
	     std::vector<int> const & flagsEE_,
	     std::vector<int> const & severitiesEB_,
	     std::vector<int> const & severitiesEE_,
	     edm::ConsumesCollector && iC);

  void calculate(const reco::Photon*, 
		 const edm::Event&, const edm::EventSetup& es,
		 reco::Photon::FiducialFlags& phofid, 
		 reco::Photon::IsolationVariables& phoisolR03, 
		 reco::Photon::IsolationVariables& phoisolR04 ) const;



private:

  static void classify(const reco::Photon* photon, 
		bool &isEBPho,
		bool &isEEPho,
		bool &isEBEtaGap,
		bool &isEBPhiGap,
		bool &isEERingGap,
		bool &isEEDeeGap,
		bool &isEBEEGap) dso_internal;



  void calculateTrackIso(const reco::Photon* photon,
			 const edm::Event &e,
			 double &trkCone,
			 int &ntrkCone,
			 double pTThresh=0,
			 double RCone=.4,
			 double RinnerCone=.1,
                         double etaSlice=0.015,
                         double lip=0.2,
                         double d0=0.1) const dso_internal;



  double calculateEcalRecHitIso(const reco::Photon* photon,
				const edm::Event& iEvent,
				const edm::EventSetup& iSetup,
				double RCone,
				double RConeInner,
                                double etaSlice,
				double eMin,
				double etMin, 
				bool vetoClusteredHits, 
				bool useNumCrystals) const dso_internal;

  double calculateHcalTowerIso(const reco::Photon* photon,
			       const edm::Event& iEvent,
			       const edm::EventSetup& iSetup,
			       double RCone,
			       double RConeInner,
			       double eMin,
                               signed int depth) const dso_internal;


  double calculateHcalTowerIso(const reco::Photon* photon,
			       const edm::Event& iEvent,
			       const edm::EventSetup& iSetup,
			       double RCone,
			       double eMin,
                               signed int depth) const dso_internal;



  
 private:

  edm::EDGetToken barrelecalCollection_;
  edm::EDGetToken endcapecalCollection_;
  edm::EDGetToken hcalCollection_;

  edm::EDGetToken trackInputTag_;
  edm::EDGetToken beamSpotProducerTag_;
  double modulePhiBoundary_;
  std::vector<double> moduleEtaBoundary_;
  bool vetoClusteredEcalHits_;
  bool useNumCrystals_;

  double trkIsoBarrelRadiusA_[6];
  double ecalIsoBarrelRadiusA_[5];
  double hcalIsoBarrelRadiusA_[9];
  double trkIsoBarrelRadiusB_[6];
  double ecalIsoBarrelRadiusB_[5];
  double hcalIsoBarrelRadiusB_[9];

  double trkIsoEndcapRadiusA_[6];
  double ecalIsoEndcapRadiusA_[5];
  double hcalIsoEndcapRadiusA_[9];
  double trkIsoEndcapRadiusB_[6];
  double ecalIsoEndcapRadiusB_[5];
  double hcalIsoEndcapRadiusB_[9];


  std::vector<int> flagsEB_;
  std::vector<int> flagsEE_;
  std::vector<int> severityExclEB_;
  std::vector<int> severityExclEE_;


};

#endif // PhotonIsolationCalculator_H
