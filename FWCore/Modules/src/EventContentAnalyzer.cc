// -*- C++ -*-
//
// Package:    Modules
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
// $Id: EventContentAnalyzer.cc,v 1.4 2005/10/07 16:44:29 marafino Exp $
//
//


// system include files
#include <iostream>
#include <iomanip>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/BranchDescription.h"
#include "FWCore/Framework/interface/Provenance.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Modules/src/EventContentAnalyzer.h"

#include "boost/lexical_cast.hpp"
//
// class declarations
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
  indentation_(iConfig.getUntrackedParameter("indentation",std::string("++")))
, evno_(0)
{
   //now do what ever initialization is needed

}

EventContentAnalyzer::~EventContentAnalyzer()
{
 
   // do anything here that needs to be done at destruction time
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
   std::string friendlyName;
   std::string modLabel;
   std::string instanceName;

   iEvent.getAllProvenance(provenances);
   
   std::cout << "\n" << indentation_ << "Event " << std::setw(5) << evno_ << " contains "
             << provenances.size() << " product" << (provenances.size()==1 ?"":"s")
             << " with friendlyClassName, moduleLabel and productInstanceName:"
             << std::endl;

   for(Provenances::iterator itProv = provenances.begin();
                             itProv != provenances.end();
                           ++itProv) {
      friendlyName = (*itProv)->product.friendlyClassName_;
      modLabel = (*itProv)->product.module.moduleLabel_;
      instanceName = (*itProv)->product.productInstanceName_;
      
      std::cout << indentation_ << friendlyName
                << " " << modLabel
                << " " << instanceName << std::endl;
      ++cumulates_["product.friendlyClassName_="+friendlyName];
      ++cumulates_["product.module.moduleLabel_="+modLabel];
      ++cumulates_["product.productInstanceName_="+instanceName];

   }
   ++evno_;
}

// ------------ method called at end of job -------------------
void
EventContentAnalyzer::endJob() 
{
   typedef std::map<std::string,int> nameMap;

   std::cout <<"\nSummary:" << std::endl;
   for(nameMap::const_iterator it =cumulates_.begin();
                               it!=cumulates_.end();
                             ++it) {
//    std::cout << "   Key " << it->first << " occurs " << it->second << " times." << std::endl;
      std::cout << std::setw(6) << it->second << " occurrences of key " << it->first << std::endl;
   }

// Test boost::lexical_cast  We don't need this right now so comment it out.
// int k = 137;
// std::string ktext = boost::lexical_cast<std::string>(k);
// std::cout << "\nInteger " << k << " expressed as a string is |" << ktext << "|" << std::endl;
}
