// -*- C++ -*-
//
// Package:    HLTSimpleJet
// Class:      HLTSimpleJet
// 
/**\class HLTSimpleJet HLTSimpleJet.cc HLTrigger/HLTcore/src/HLTSimpleJet.cc

 Description: A very basic HLT trigger for jets

 Implementation:
     A filter is provided cutting on the number of jets above a pt cut
*/
//
// Original Author:  Martin GRUNEWALD
//         Created:  Thu Mar 23 10:00:22 CET 2006
// $Id: HLTSimpleJet.cc,v 1.1 2006/03/23 15:11:09 gruen Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/JetObjects/interface/CaloJetCollection.h"
#include "DataFormats/JetObjects/interface/CaloJet.h"

//
// class decleration
//

class HLTSimpleJet : public edm::EDFilter {

   public:
      explicit HLTSimpleJet(const edm::ParameterSet&);
      ~HLTSimpleJet();

      virtual bool filter(const edm::Event&, const edm::EventSetup&);

   private:
      // ----------member data ---------------------------

      std::string module_;  // module label for input jets
      double ptcut_;        // pt cut in GeV 
      int    njcut_;        // number of jets required
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HLTSimpleJet::HLTSimpleJet(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
   module_ = iConfig.getParameter< std::string > ("input");
   ptcut_  = iConfig.getParameter<double> ("ptcut");
   njcut_  = iConfig.getParameter<int> ("njcut");
   std::cout << "HLTSimpleJet input: " << module_ << std::endl;
   std::cout << "             PTcut: " << ptcut_  << std::endl;
   std::cout << "    Number of jets: " << njcut_  << std::endl;
}


HLTSimpleJet::~HLTSimpleJet()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

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
