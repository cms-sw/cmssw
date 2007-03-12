#ifndef MCATNLO_SOURCE_EMPTY_FILTER_H
#define MCATNLO_SOURCE_EMPTY_FILTER_H

//
// Original Author:  Fabian Stoeckli
//         Created:  Mon Mar 12 17:34:14 CET 2007
// $Id$
//
//

// Filter to remove empty events produced with MC@NLO/HERWIG


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

class EmptyFilter : public edm::EDFilter {
   public:
      explicit EmptyFilter(const edm::ParameterSet&);
      ~EmptyFilter();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
};

#endif
