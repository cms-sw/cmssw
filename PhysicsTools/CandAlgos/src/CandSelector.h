#ifndef CANDCOMBINER_CANDSELECTOR_H
#define CANDCOMBINER_CANDSELECTOR_H
// -*- C++ -*-
//
// Package:     CandCombiner
// Class  :     CandSelector
// 
/**\class CandSelector CandSelector.h PhysicsTools/CandCombiner/interface/CandSelector.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Thu Aug 11 16:50:33 EDT 2005
// $Id$
//

// system include files
#include "boost/shared_ptr.hpp"
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

// forward declarations
namespace aod {
  class Selector;
}

class CandSelector : public edm::EDProducer {
   public:
      explicit CandSelector( const edm::ParameterSet& );
      ~CandSelector();

      virtual void produce( edm::Event&, const edm::EventSetup& );
   private:
      // ----------member data ---------------------------
    std::string src_;
    boost::shared_ptr<aod::Selector> pSelect_;
};


#endif /* CANDCOMBINER_CANDSELECTOR_H */
