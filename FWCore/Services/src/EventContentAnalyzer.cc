// -*- C++ -*-
//
// Package:    Services
// Class:      EventContentAnalyzer
// 
/**
 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chris Jones
//         Created:  Mon Sep 19 11:47:28 CEST 2005
// $Id: EventContentAnalyzer.cc,v 1.1 2005/09/19 12:16:05 chrjones Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/BranchDescription.h"
#include "FWCore/Framework/interface/Provenance.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Services/src/EventContentAnalyzer.h"

//
// class decleration
//

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
EventContentAnalyzer::EventContentAnalyzer(const edm::ParameterSet& iConfig) :
indentation_( iConfig.getUntrackedParameter("indentation",std::string( "++") ) )
{
   //now do what ever initialization is needed

}


EventContentAnalyzer::~EventContentAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
EventContentAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   typedef std::vector< Provenance const*> Provenances;
   Provenances provenances;
   iEvent.getAllProvenance( provenances );
   
   std::cout <<indentation_<<"Event contains "<<provenances.size()<< " product"<< (provenances.size()==1 ?"":"s")<<std::endl;
   for(Provenances::iterator itProv = provenances.begin();
       itProv != provenances.end();
       ++itProv ) {
      std::cout <<indentation_<<(*itProv)->product.friendlyClassName_
      <<" "<<(*itProv)->product.module.moduleLabel_
      <<" "<<(*itProv)->product.productInstanceName_ <<std::endl;
   }
}
