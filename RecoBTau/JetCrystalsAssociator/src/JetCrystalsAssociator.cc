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
#include "FWCore/ParameterSet/interface/InputTag.h"
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

#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

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
      ~JetCrystalsAssociator();


      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
     JetCrystalsAssociationCollection * associate( const edm::Handle<CaloJetCollection> & jets,
						   const edm::OrphanHandle<EMLorentzVectorCollection>   & myLorentzRecHits) const;


  // ----------member data ---------------------------
  edm::InputTag m_jetsSrc;
  edm::InputTag m_towersSrc;
  double m_deltaRCut;
};

JetCrystalsAssociator::JetCrystalsAssociator(const edm::ParameterSet& iConfig)
{
  produces<reco::JetCrystalsAssociationCollection>();
  produces<reco::EMLorentzVectorCollection>();

  m_towersSrc = iConfig.getParameter<edm::InputTag>("towers");
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
  iSetup.get<IdealGeometryRecord>().get(geometry);
  
  const CaloSubdetectorGeometry* EB = geometry->getSubdetectorGeometry(DetId::Ecal,EcalBarrel);
   const CaloSubdetectorGeometry* EE = geometry->getSubdetectorGeometry(DetId::Ecal,EcalEndcap);
   // end 
   
   Handle<CaloJetCollection> jets;
   iEvent.getByLabel(m_jetsSrc, jets);
   
   // get calo towers collection
   Handle<CaloTowerCollection> caloTowers; 
   iEvent.getByLabel(m_towersSrc, caloTowers);

   // calculation of ECAL isolation
   Handle<EBRecHitCollection> EBRecHits;
   Handle<EERecHitCollection> EERecHits;
   iEvent.getByLabel( "ecalRecHit", "EcalRecHitsEB", EBRecHits );
   iEvent.getByLabel( "ecalRecHit", "EcalRecHitsEE", EERecHits );
   
   EMLorentzVectorCollection*  myLorentzRecHits = new EMLorentzVectorCollection();
   //loop on jets and associate
  for (size_t t = 0; t < jets->size(); t++)
    {
      const std::vector<CaloTowerDetId>&  detIDs=(*jets)[t].getTowerIndices();
      int nConstituents= detIDs.size();
      // access towers which belong to jet
      for (int i = 0; i <nConstituents ; i++) {
	//Find the tower from its CaloTowerDetID	
	CaloTowerCollection::const_iterator theTower=caloTowers->find(detIDs[i]);
	//	if(theTower != caloTowers->end()) continue;
	int ietaTower = detIDs[i].ieta();
	int iphiTower = detIDs[i].iphi();
	size_t numRecHits = theTower->constituentsSize();
	// access CaloRecHits
	for(size_t j = 0; j <numRecHits ; j++) {
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
		if(!this_cell){
		}else{
		  GlobalPoint posi = this_cell->getPosition();
		  double energy = theRecHit->energy();
		  		  double eta = posi.eta();
		  double phi = posi.phi();
		  double theta = posi.theta();
		  if(theta > 3.14159) theta = 2*3.14159 - theta;
		  double et = energy * sin(theta);
		  EMLorentzVector p(et, eta, phi, energy);
		  myLorentzRecHits->push_back(p);
		}
	      }
	    }else if(  EcalNum == 2 ){
	      EEDetId EcalID = RecHitDetID;
	      EERecHitCollection::const_iterator theRecHit=EERecHits->find(EcalID);	    
	      if(theRecHit != EBRecHits->end()){
		DetId id = theRecHit->detid();
		const CaloCellGeometry* this_cell = EE->getGeometry(id);
		if(!this_cell){
		}else{
		  GlobalPoint posi = this_cell->getPosition();
		  		  double energy = theRecHit->energy();
		  double eta = posi.eta();
		  double phi = posi.phi();
		  double theta = posi.theta();
		  if(theta > 3.14159) theta = 2*3.14159 - theta;
		  double et = energy * sin(theta);
		  EMLorentzVector p(et, eta, phi, energy);
		  myLorentzRecHits->push_back(p);
		}
	      }
	    }
	  }
	}
      }
    }



  std::auto_ptr<EMLorentzVectorCollection> jetRecHits(myLorentzRecHits);
  edm::OrphanHandle <reco::EMLorentzVectorCollection >  myRecHits =  iEvent.put(jetRecHits);
  //  iEvent.put(jetRecHits);

  std::auto_ptr<JetCrystalsAssociationCollection> jetCrystals(associate(jets,myRecHits));
  iEvent.put(jetCrystals);
}

JetCrystalsAssociationCollection * JetCrystalsAssociator::associate( const edm::Handle<CaloJetCollection> & jets,
                                                                 const edm::OrphanHandle<EMLorentzVectorCollection> & myLorentzRecHits  ) const
{
  JetCrystalsAssociationCollection * outputCollection = new JetCrystalsAssociationCollection();

  //loop on jets and associate
  for (size_t j = 0; j < jets->size(); j++)
    {
      for (size_t t =0 ; t < myLorentzRecHits->size();t++)
	{
	  
	  double delta  = ROOT::Math::VectorUtil::DeltaR((*jets)[j].p4().Vect(), (*myLorentzRecHits)[t]);
	  if(delta < m_deltaRCut)
	    outputCollection->insert(edm::Ref<CaloJetCollection>(jets, j), edm::Ref<EMLorentzVectorCollection>(myLorentzRecHits, t));
	}
    }  
  return outputCollection;
}

//define this as a plug-in
DEFINE_FWK_MODULE(JetCrystalsAssociator);
