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
// $Id: EcalIsolation.cc,v 1.5 2007/05/13 22:06:27 gennai Exp $
//
//
#include <string>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTauTag/EcalIsolation/interface/EcalIsolation.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"
#include "DataFormats/BTauReco/interface/EMIsolatedTauTagInfo.h"

// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

//
// constructors and destructor
//
EcalIsolation::EcalIsolation(const edm::ParameterSet& iConfig):
  mJetForFilter(iConfig.getParameter<edm::InputTag>("JetForFilter")),
  mSmallCone(iConfig.getParameter<double>("SmallCone")),
  mBigCone(iConfig.getParameter<double>("BigCone")),
  pIsolCut(iConfig.getParameter<double>("Pisol")){
    

   produces<reco::EMIsolatedTauTagInfoCollection>();

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
   Handle<JetCrystalsAssociationCollection> jets;
   JetCrystalsAssociationCollection::const_iterator CI;
   iEvent.getByLabel(mJetForFilter, jets);
   CI = (jets.product())->begin();
   
   EMIsolatedTauTagInfoCollection* myColl = new EMIsolatedTauTagInfoCollection();
   

     int i =0;
   
   for(;CI!=(jets.product())->end();CI++){
     EMIsolatedTauTagInfo myTag(0,edm::Ref<JetCrystalsAssociationCollection>(jets,i));
     double discriminator = myTag.discriminator(mBigCone,mSmallCone,pIsolCut);
     myTag.setDiscriminator(discriminator);
     myColl->push_back(myTag);
     i++;
   }
   auto_ptr<EMIsolatedTauTagInfoCollection> myEMIsolTaus(myColl);
   iEvent.put(myEMIsolTaus);

}



//define this as a plug-in
DEFINE_FWK_MODULE(EcalIsolation);
