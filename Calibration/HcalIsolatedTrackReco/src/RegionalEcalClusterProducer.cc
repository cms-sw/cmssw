// -*- C++ -*-
//
// Package:    RegionalEcalClusterProducer
// Class:      RegionalEcalClusterProducer
// 
/**\class RegionalEcalClusterProducer RegionalEcalClusterProducer.cc Calibration/RegionalEcalClusterProducer/src/RegionalEcalClusterProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Grigory Safronov
//         Created:    1 17:08:06 MSD 2007
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Calibration/HcalIsolatedTrackReco/interface/RegionalEcalClusterProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"



//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
RegionalEcalClusterProducer::RegionalEcalClusterProducer(const edm::ParameterSet& conf)
{
  searchAroundTrack_=conf.getParameter<bool>("SearchAroundTrack");
  deltaEtaSearch_=conf.getParameter<double>("DeltaEtaSearch");
  deltaPhiSearch_=conf.getParameter<double>("DeltaPhiSearch");
  useEndcap_=conf.getParameter<bool>("UseEndcap");

  EBRecHitCollectionLabel_=conf.getParameter<edm::InputTag>("EBRecHitCollectionLabel");
  EERecHitCollectionLabel_=conf.getParameter<edm::InputTag>("EERecHitCollectionLabel");
  barrelClusterCollection_=conf.getParameter<std::string>("BarrelClusterCollection");
  endcapClusterCollection_=conf.getParameter<std::string>("EndcapClusterCollection");

  barrelSeedThresh_=conf.getParameter<double>("BarrelSeedThreshold");
  endcapSeedThresh_=conf.getParameter<double>("EndcapSeedThreshold");
  l1tausource_=conf.getParameter<edm::InputTag>("L1TauSource");
  prodtracksource_=conf.getParameter<edm::InputTag>("TracksSource");


  std::map<std::string,double> providedParameters;
  providedParameters.insert(std::make_pair("LogWeighted",true));
  providedParameters.insert(std::make_pair("T0_barl",7.4));
  providedParameters.insert(std::make_pair("T0_endc",3.1));
  providedParameters.insert(std::make_pair("T0_endcPresh",1.2));
  providedParameters.insert(std::make_pair("W0",4.2));
  providedParameters.insert(std::make_pair("X0",0.89));
  PositionCalc posCalculator_ = PositionCalc(providedParameters);

  islandAlg=new IslandClusterAlgo(barrelSeedThresh_, endcapSeedThresh_,posCalculator_,IslandClusterAlgo::pINFO);

  produces< reco::BasicClusterCollection >(endcapClusterCollection_);
  produces< reco::BasicClusterCollection >(barrelClusterCollection_);

}


RegionalEcalClusterProducer::~RegionalEcalClusterProducer()

{
  delete islandAlg;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
RegionalEcalClusterProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  std::vector<EcalEtaPhiRegion> regionsBar;
  std::vector<EcalEtaPhiRegion> regionsEc;
  
  if (!searchAroundTrack_)
    {
      edm::Handle<l1extra::L1JetParticleCollection> l1taus;
      try {iEvent.getByLabel(l1tausource_,l1taus);} catch (...) {edm::LogError("FAIL:")<<"-- No l1 taus";} 
      for (l1extra::L1JetParticleCollection::const_iterator tit=l1taus->begin(); tit!=l1taus->end(); tit++)
	{
	  if (fabs(tit->eta())+deltaEtaSearch_<1.497) 
	    {
	      EcalEtaPhiRegion etaPhiReg(tit->eta()-deltaEtaSearch_, tit->eta()+deltaEtaSearch_, tit->phi()-deltaPhiSearch_, tit->phi()+deltaPhiSearch_);
	      regionsBar.push_back(etaPhiReg);
	    }
	  if (fabs(tit->eta())-deltaEtaSearch_<1.497&&fabs(tit->eta())+deltaEtaSearch_>1.497)
	    {
	      EcalEtaPhiRegion etaPhiReg1(tit->eta()-deltaEtaSearch_, fabs(tit->eta())*1.497/tit->eta(), tit->phi()-deltaPhiSearch_, tit->phi()+deltaPhiSearch_);
	      EcalEtaPhiRegion etaPhiReg2(fabs(tit->eta())*1.497/tit->eta(), tit->eta()+deltaEtaSearch_, tit->phi()-deltaPhiSearch_, tit->phi()+deltaPhiSearch_);
	      if (tit->eta()>0)
		{
		  regionsBar.push_back(etaPhiReg1);
		  regionsEc.push_back(etaPhiReg2);
		}
	      else 
		{
		  regionsBar.push_back(etaPhiReg2);
		  regionsEc.push_back(etaPhiReg1);
		}
	    }
	  if (fabs(tit->eta())-deltaEtaSearch_>1.497) 
	    {
	      EcalEtaPhiRegion etaPhiReg(tit->eta()-deltaEtaSearch_, tit->eta()+deltaEtaSearch_, tit->phi()-deltaPhiSearch_, tit->phi()+deltaPhiSearch_);
	      regionsEc.push_back(etaPhiReg);
	    }
			      
	}
    }
  else
    {
      edm::Handle<reco::IsolatedPixelTrackCandidateCollection> prodTracks;
      try {iEvent.getByLabel(prodtracksource_,prodTracks);} catch (...) {edm::LogError("FAIL:")<<"-- No pixelTracks";}
      
      for (reco::IsolatedPixelTrackCandidateCollection::const_iterator pit=prodTracks->begin(); pit!=prodTracks->end(); pit++)
	{
	  if (fabs(pit->eta())+deltaEtaSearch_<1.497) 
	    {
	      EcalEtaPhiRegion etaPhiReg(pit->eta()-deltaEtaSearch_, pit->eta()+deltaEtaSearch_, pit->phi()-deltaPhiSearch_, pit->phi()+deltaPhiSearch_);
	      regionsBar.push_back(etaPhiReg);
	    }
	  if (fabs(pit->eta())-deltaEtaSearch_<1.497&&fabs(pit->eta())+deltaEtaSearch_>1.497)
	    {
	      EcalEtaPhiRegion etaPhiReg1(pit->eta()-deltaEtaSearch_, fabs(pit->eta())*1.497/pit->eta(), pit->phi()-deltaPhiSearch_, pit->phi()+deltaPhiSearch_);
	      EcalEtaPhiRegion etaPhiReg2(fabs(pit->eta())*1.497/pit->eta(), pit->eta()+deltaEtaSearch_, pit->phi()-deltaPhiSearch_, pit->phi()+deltaPhiSearch_);
	      if (pit->eta()>0)
		{
		  regionsBar.push_back(etaPhiReg1);
		  regionsEc.push_back(etaPhiReg2);
		}
	      else 
		{
		  regionsBar.push_back(etaPhiReg2);
		  regionsEc.push_back(etaPhiReg1);
		}
	    }
	  if (fabs(pit->eta())-deltaEtaSearch_>1.497) 
	    {
	      EcalEtaPhiRegion etaPhiReg(pit->eta()-deltaEtaSearch_, pit->eta()+deltaEtaSearch_, pit->phi()-deltaPhiSearch_, pit->phi()+deltaPhiSearch_);
	      regionsEc.push_back(etaPhiReg);
	    }
		
	}
    }
  clusterize(iEvent, iSetup, EERecHitCollectionLabel_, EBRecHitCollectionLabel_, barrelClusterCollection_, regionsBar, IslandClusterAlgo::barrel);
  if (useEndcap_) clusterize(iEvent, iSetup, EERecHitCollectionLabel_, EBRecHitCollectionLabel_, endcapClusterCollection_, regionsEc, IslandClusterAlgo::endcap);

}

// ------------ method called once each job just before starting event loop  ------------
void 
RegionalEcalClusterProducer::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------

void 
RegionalEcalClusterProducer::endJob() {
}



void RegionalEcalClusterProducer::clusterize(edm::Event &iEvent, const edm::EventSetup &iSetup, edm::InputTag EERecHit_, edm::InputTag EBRecHit_, std::string cluColl_, std::vector<EcalEtaPhiRegion> etaphi, IslandClusterAlgo::EcalPart ecal_part)
{

  edm::Handle<EcalRecHitCollection> ecalRHEE;
  iEvent.getByLabel(EERecHit_, ecalRHEE);
  
  edm::Handle<EcalRecHitCollection> ecalRHEB;
  iEvent.getByLabel(EBRecHit_, ecalRHEB);
  
  //access to ecal geometry

  edm::ESHandle<CaloGeometry> caloG;
  iSetup.get<IdealGeometryRecord>().get(caloG);
  
  const CaloSubdetectorGeometry *geometry_p;
  CaloSubdetectorTopology *topology_p;

  const EcalRecHitCollection *ecalRecHit;

  if (ecal_part==IslandClusterAlgo::barrel)
    {
      geometry_p = caloG->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
      topology_p = new EcalBarrelTopology(caloG);
      ecalRecHit=ecalRHEB.product();
    }
  else 
    {
      geometry_p = caloG->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
      topology_p = new EcalEndcapTopology(caloG);
      ecalRecHit=ecalRHEE.product();
    }
  
  const CaloSubdetectorGeometry *geometryES_p;
  geometryES_p = caloG->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);
  
  reco::BasicClusterCollection clusters;
  bool regional=true;

  clusters = islandAlg->makeClusters(ecalRecHit, geometry_p, topology_p, geometryES_p, ecal_part, regional, etaphi);

  std::auto_ptr<reco::BasicClusterCollection> pOut(new reco::BasicClusterCollection);
  pOut->assign(clusters.begin(), clusters.end());
  edm::OrphanHandle<reco::BasicClusterCollection> bccHandle;
  bccHandle = iEvent.put(pOut, cluColl_);

  delete topology_p;
}

