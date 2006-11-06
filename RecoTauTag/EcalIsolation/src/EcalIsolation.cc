// -*- C++ -*-
//
// Package:    EcalIsolation
// Class:      EcalIsolation
// 
/**\class EcalIsolation EcalIsolation.cc RecoTauTag/EcalIsolation/src/EcalIsolation.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Artur Kalinowski
//         Created:  Mon Sep 11 12:48:02 CEST 2006
// $Id$
//
//
#include <string>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTauTag/EcalIsolation/interface/EcalIsolation.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"
#include "DataFormats/BTauReco/interface/JetEisolAssociation.h"

// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

//
// constructors and destructor
//
EcalIsolation::EcalIsolation(const edm::ParameterSet& iConfig):
  mJetForFilter(iConfig.getParameter<std::string>("JetForFilter")),
  mCaloTowers(iConfig.getParameter<std::string>("CaloTowers")),
  mSmallCone(iConfig.getParameter<double>("SmallCone")),
  mBigCone(iConfig.getParameter<double>("BigCone")){
    

  produces<reco::JetEisolAssociationCollection>();

}


EcalIsolation::~EcalIsolation()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
EcalIsolation::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco;
   using namespace std;

  // get calo jetHLT collection
   Handle<CaloJetCollection> jets;
   CaloJetCollection::const_iterator CI;
   iEvent.getByLabel(mJetForFilter, jets);
   CI = jets.product()->begin();

   std::auto_ptr<JetEisolAssociationCollection> myAssoc(new JetEisolAssociationCollection());


   LogDebug("") <<"jets->size = " << jets->size() << endl;
   int i=0;
   for(;CI!=jets.product()->end();CI++){
     CaloJetRef aRef(jets,i);
          myAssoc->insert(aRef,
     		     checkIsolation(&(*(CI)),iEvent));
     i++;
   }

   iEvent.put(myAssoc);

}


float EcalIsolation::checkIsolation(const reco::CaloJet * aJet, edm::Event& iEvent){

  using namespace edm;
  using namespace reco;
  using namespace std;

  LogDebug("") << "->   Jet HLT " << " pT: " 
	       << aJet->pt()
	       << " eta: " << aJet->eta()
	       << " phi: " << aJet->phi() << endl;

  // get calo towers collection
  Handle<CaloTowerCollection> calotowers; 
  iEvent.getByLabel(mCaloTowers, calotowers);  
  LogDebug("") <<" all calo towers = " << calotowers->size() << endl;
                           
  int nconstituents = aJet->n90();
  LogDebug("") <<" Number of jet 90% constituents (towers) = " << nconstituents << endl;
  const vector<CaloTowerDetId> calodetid = aJet->getTowerIndices();

  int icount0 = 0;
  int icountu = 0;
  double Pisol = 0.;
       
  for (vector<CaloTowerDetId>::const_iterator it=calodetid.begin();it!=calodetid.end();it++)
    {
      CaloTowerCollection::const_iterator ict = calotowers->find(*it);
      if(ict != calotowers->end()) {
	
	math::PtEtaPhiELorentzVector p( (*ict).et(), (*ict).eta(), (*ict).phi(), (*ict).energy() );
	double delta  = ROOT::Math::VectorUtil::DeltaR(aJet->p4().Vect(), p);
	     
	icount0 += 1;
	LogDebug("") <<" icount0 = " << icount0 <<" delta = " << delta << endl;

	LogDebug("") <<" icountu = " << icountu
		     <<" Et = " << (*ict).et()
		     <<" Number of constituents "<< (*ict).constituentsSize()
		     <<" eta = " << (*ict).eta()
		     <<" phi = " << (*ict).phi()
		     <<" emE = " << (*ict).emEnergy()
		     <<" emEt = " << (*ict).emEt()
		     <<" hadEt = " << (*ict).hadEt()
		     <<" Pisol = " << Pisol
		     <<" delta = " << delta << endl;
	     
	if(delta > mSmallCone && delta < mBigCone) {
	       
	  Pisol += (*ict).emEt();  
/*	       
	       icountu += 1;
	       cout <<" icountu = " << icountu 
		    <<" Et = " << (*ict).et()
		    <<" Number of constituents "<< (*ict).constituentsSize()
		    <<" eta = " << (*ict).eta()
		    <<" phi = " << (*ict).phi()
		    <<" emE = " << (*ict).emEnergy()
		    <<" emEt = " << (*ict).emEt()
		    <<" hadEt = " << (*ict).hadEt() 
		    <<" Pisol = " << Pisol 
		    <<" delta = " << delta << endl;
*/
	}
      }
    }
  LogDebug("")<<"Pisol: "<<Pisol<<endl; 
  return Pisol;
}


//define this as a plug-in
DEFINE_FWK_MODULE(EcalIsolation);
