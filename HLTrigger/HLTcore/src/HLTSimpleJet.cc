// -*- C++ -*-
//
// see header file for documentation
//
// $Id: HLTSimpleJet.cc,v 1.7 2006/04/26 09:55:34 gruen Exp $
//

#include "HLTrigger/HLTcore/interface/HLTSimpleJet.h"

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

//
// constructors and destructor
//
HLTSimpleJet::HLTSimpleJet(const edm::ParameterSet& iConfig)
{
   module_ = iConfig.getParameter< std::string > ("input");
   ptcut_  = iConfig.getParameter<double> ("ptcut");
   njcut_  = iConfig.getParameter<int> ("njcut");
   std::cout << "HLTSimpleJet input: " << module_ << std::endl;
   std::cout << "             PTcut: " << ptcut_  << std::endl;
   std::cout << "    Number of jets: " << njcut_  << std::endl;

   //register your products
   produces<reco::HLTFilterObjectWithRefs>();
}

HLTSimpleJet::~HLTSimpleJet()
{
   std::cout << "HLTSimpleJet destroyed! " << std::endl;
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool
HLTSimpleJet::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;

   //   cout << "HLTSimpleJet::filter start:" << endl;

   Handle<CaloJetCollection>  jets;
   iEvent.getByLabel (module_,jets);

   auto_ptr<reco::HLTFilterObjectWithRefs> filterproduct (new reco::HLTFilterObjectWithRefs);

   int n=0;
   CaloJetCollection::const_iterator jet(jets->begin());
   for (; jet!=jets->end()&&n<njcut_; jet++) {
     //     cout << (*jet).pt() << endl;
     if ( (jet->pt()) >= ptcut_) {
       n++;
       filterproduct->putJet(Ref<CaloJetCollection>(jets,distance(jets->begin(),jet)));
     }
   }

   bool accept(n>=njcut_);
   filterproduct->setAccept(accept);
   iEvent.put(filterproduct);

   //   std::cout << "HLTSimpleJet::filter stop: " << n << std::endl;

   return accept;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HLTSimpleJet)
