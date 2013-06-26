// -*- C++ -*-
//
// Package:    LQGenFilter
// Class:      LQGenFilter
// 
/**\class LQGenFilter LQGenFilter.cc GeneratorInterface/GenFilters/src/LQGenFilter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Alexey Ferapontov,8 R-021,+41227676332,
//         Created:  Mon Mar  8 15:28:06 CET 2010
// $Id: LQGenFilter.h,v 1.1 2010/03/23 13:26:26 fabstoec Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

//
// class declaration
//

class LQGenFilter : public edm::EDFilter {
   public:
      explicit LQGenFilter(const edm::ParameterSet&);
      ~LQGenFilter();

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
      edm::InputTag src_;
      bool eejj_, enuejj_, nuenuejj_, mumujj_, munumujj_, numunumujj_; 
};
