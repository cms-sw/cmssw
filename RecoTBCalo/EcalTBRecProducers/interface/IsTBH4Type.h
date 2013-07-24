// -*- C++ -*-
//
// Package:    IsTBH4Type
// Class:      IsTBH4Type
// 
/**\class IsTBH4Type IsTBH4Type.cc RecoTBCalo/IsTBH4Type/src/IsTBH4Type.cc

 Description: tag a given type of run

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Pietro Govoni
//         Created:  Thu Aug 10 16:21:22 CEST 2006
// $Id: IsTBH4Type.h,v 1.1 2006/08/15 09:32:34 govoni Exp $
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

//
// class declaration
//

class IsTBH4Type : public edm::EDFilter {
   public:
      explicit IsTBH4Type(const edm::ParameterSet&);
      ~IsTBH4Type();

      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:

      // ----------member data ---------------------------
      
   //! collection of the event header
   std::string eventHeaderCollection_ ;
   //! producer of the event header
   std::string eventHeaderProducer_ ;
   //! type of run to flag
   std::string typeToFlag_ ;
   //! what to return in case no header is found
   bool notFound_ ;

};

