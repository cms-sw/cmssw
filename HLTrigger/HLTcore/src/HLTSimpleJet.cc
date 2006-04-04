// -*- C++ -*-
//
// see header file for documentation
//
// $Id: HLTSimpleJet.cc,v 1.3 2006/03/27 14:42:42 gruen Exp $
//

#include "HLTrigger/HLTcore/interface/HLTSimpleJet.h"

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

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
HLTSimpleJet::filter(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   //   std::cout << "HLTSimpleJet::filter start:" << std::endl;

   Handle<CaloJetCollection> jets;
   iEvent.getByLabel (module_, jets);

   CaloJetCollection::const_iterator jet = jets->begin ();

   int n=0;
   for (; jet != jets->end (); jet++) {
     //     std::cout << (*jet).getPt() << std::endl;
     if ( (*jet).getPt() >= ptcut_) n++;
   }
   //   std::cout << "HLTSimpleJet::filter stop: " << n << std::endl;

   return (n>=njcut_);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HLTSimpleJet)
