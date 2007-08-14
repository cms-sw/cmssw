#ifndef POMWIG_SOURCE_EMPTY_FILTER_H
#define POMWIG_SOURCE_EMPTY_FILTER_H

//
// Original Author:  Fabian Stoeckli
//         Created:  Mon Mar 12 17:34:14 CET 2007
//         (Herwig6Interface)
//
// Modified for PomwigInterface: Antonio.Vilela.Pereira@cern.ch
//
// $Id: Herwig6Filter.h,v 1.1 2007/03/15 10:19:11 fabstoec Exp $
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

class PomwigFilter : public edm::EDFilter {
   public:
      explicit PomwigFilter(const edm::ParameterSet&);
      ~PomwigFilter();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
};

#endif
