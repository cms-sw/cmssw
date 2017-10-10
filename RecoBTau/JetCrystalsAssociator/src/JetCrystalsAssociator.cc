// -*- C++ -*-
//
// Package:    JetCrystalsAssociator
// Class:      JetCrystalsAssociator
// 
/**\class JetCrystalsAssociator JetCrystalsAssociator.cc RecoBTag/JetCrystalsAssociator/src/JetCrystalsAssociator.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Simone Gennai
//         Created:  Wed Apr 12 11:12:49 CEST 2006
//
//


// system include files
#include <memory>
#include <string>
#include <iostream>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESHandle.h"


#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/BTauReco/interface/JetCrystalsAssociation.h"

//Calorimeter stuff
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h" 
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"
using namespace reco;

//
// class decleration
//

class JetCrystalsAssociator : public edm::EDProducer {

   public:
      explicit JetCrystalsAssociator(const edm::ParameterSet&);
      ~JetCrystalsAssociator() override;


      void produce(edm::Event&, const edm::EventSetup&) override;
   private:
      std::unique_ptr<JetCrystalsAssociationCollection> associate( 
          const edm::Handle<CaloJetCollection> & jets,
	  const edm::OrphanHandle<EMLorentzVectorCollection> & myLorentzRecHits) const;


  // ----------member data ---------------------------
  edm::InputTag m_jetsSrc;
  edm::InputTag m_EBRecHits;
  edm::InputTag m_EERecHits;

  double m_deltaRCut;
};

JetCrystalsAssociator::JetCrystalsAssociator(const edm::ParameterSet& iConfig)
{
  produces<reco::JetCrystalsAssociationCollection>();
  produces<reco::EMLorentzVectorCollection>();

  m_EBRecHits = iConfig.getParameter<edm::InputTag>("EBRecHits");
  m_EERecHits = iConfig.getParameter<edm::InputTag>("EERecHits");
  m_jetsSrc   = iConfig.getParameter<edm::InputTag>("jets");
  m_deltaRCut = iConfig.getParameter<double>("coneSize");
}


JetCrystalsAssociator::~JetCrystalsAssociator()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
JetCrystalsAssociator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace reco;
  using namespace std;

  // geometry initialization
  ESHandle<CaloGeometry> geometry;
  iSetup.get<CaloGeometryRecord>().get(geometry);
  
  const CaloSubdetectorGeometry* EB = geometry->getSubdetectorGeometry(DetId::Ecal,EcalBarrel);
   const CaloSubdetectorGeometry* EE = geometry->getSubdetectorGeometry(DetId::Ecal,EcalEndcap);
   // end 
   
   Handle<CaloJetCollection> jets;
   iEvent.getByLabel(m_jetsSrc, jets);
   
   // get calo towers collection
   //   Handle<CaloTowerCollection> caloTowers; 
   //   iEvent.getByLabel(m_towersSrc, caloTowers);

   // calculation of ECAL isolation
   Handle<EBRecHitCollection> EBRecHits;
   Handle<EERecHitCollection> EERecHits;
   iEvent.getByLabel( m_EBRecHits, EBRecHits );
   iEvent.getByLabel( m_EERecHits, EERecHits );
   
   auto jetRecHits = std::make_unique<EMLorentzVectorCollection>();
   //loop on jets and associate
   for (size_t t = 0; t < jets->size(); t++)
    {
      const std::vector<CaloTowerPtr>  myTowers=(*jets)[t].getCaloConstituents();
      //      cout <<"Jet id "<<t<<endl;
      //      cout <<"Tower size "<<myTowers.size()<<endl;
      for (unsigned int iTower = 0; iTower < myTowers.size(); iTower++)
	{
	  CaloTowerPtr theTower = myTowers[iTower];
	  size_t numRecHits = theTower->constituentsSize();
	// access CaloRecHits
	for (size_t j = 0; j < numRecHits; j++) {
	  DetId RecHitDetID=theTower->constituent(j);
	  DetId::Detector DetNum=RecHitDetID.det();
	  if( DetNum == DetId::Ecal ){
	    int EcalNum =  RecHitDetID.subdetId();
	    if( EcalNum == 1 ){
	      EBDetId EcalID = RecHitDetID;
	      EBRecHitCollection::const_iterator theRecHit=EBRecHits->find(EcalID);
	      if(theRecHit != EBRecHits->end()){
		DetId id = theRecHit->detid();
		const CaloCellGeometry* this_cell = EB->getGeometry(id);
		if (this_cell) {
		  const GlobalPoint& posi = this_cell->getPosition();
		  double energy = theRecHit->energy();
		  double eta = posi.eta();
		  double phi = posi.phi();
		  double theta = posi.theta();
		  if(theta > M_PI) theta = 2 * M_PI- theta;
		  double et = energy * sin(theta);
		  // cout <<"Et "<<et<<endl;
		  EMLorentzVector p(et, eta, phi, energy);
		  jetRecHits->push_back(p);
		}
	      }
	    } else if ( EcalNum == 2 ) {
	      EEDetId EcalID = RecHitDetID;
	      EERecHitCollection::const_iterator theRecHit=EERecHits->find(EcalID);	    
	      if(theRecHit != EBRecHits->end()){
		DetId id = theRecHit->detid();
		const CaloCellGeometry* this_cell = EE->getGeometry(id);
		if (this_cell) {
		  const GlobalPoint& posi = this_cell->getPosition();
		  double energy = theRecHit->energy();
		  double eta = posi.eta();
		  double phi = posi.phi();
		  double theta = posi.theta();
		  if (theta > M_PI) theta = 2 * M_PI - theta;
		  double et = energy * sin(theta);
		  // cout <<"Et "<<et<<endl;
		  EMLorentzVector p(et, eta, phi, energy);
		  jetRecHits->push_back(p);
		}
	      }
	    }
	  }
	}
      }
    }

  edm::OrphanHandle <reco::EMLorentzVectorCollection> myRecHits = iEvent.put(std::move(jetRecHits));

  iEvent.put(associate(jets,myRecHits));
}

std::unique_ptr<JetCrystalsAssociationCollection> JetCrystalsAssociator::associate( 
        const edm::Handle<CaloJetCollection> & jets,
        const edm::OrphanHandle<EMLorentzVectorCollection> & myLorentzRecHits) const
{
  // we know we will save an element per input jet
  auto outputCollection = std::make_unique<JetCrystalsAssociationCollection>(jets->size());

  //loop on jets and associate
  for (size_t j = 0; j < jets->size(); j++) {
    (*outputCollection)[j].first = edm::RefToBase<Jet>(CaloJetRef(jets, j));
    for (size_t t = 0; t < myLorentzRecHits->size(); t++) {
      double delta = ROOT::Math::VectorUtil::DeltaR((*jets)[j].p4().Vect(), (*myLorentzRecHits)[t]);
      if (delta < m_deltaRCut)
        (*outputCollection)[j].second.push_back( EMLorentzVectorRef(myLorentzRecHits, t) );
    }
  }  
  return outputCollection;
}

//define this as a plug-in
DEFINE_FWK_MODULE(JetCrystalsAssociator);
