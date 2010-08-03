#ifndef PhysicsTools_TagAndProbe_TagProbeEDMFilter_h
#define PhysicsTools_TagAndProbe_TagProbeEDMFilter_h
// -*- C++ -*-
//
// Package:     TagAndProbe
// Class  :     TagProbeEDMFilter
// 
/**\class TagProbeEDMFilter TagProbeEDMFilter.h PhysicsTools/TagAndProbe/interface/TagProbeEDMFilter.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author: Nadia Adam (Princeton) 
//         Created:  Fri Jun  6 09:13:10 CDT 2008
// $Id: TagProbeEDMFilter.h,v 1.2 2008/07/30 13:38:24 srappocc Exp $
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

class TagProbeEDMFilter : public edm::EDFilter 
{
   public:
      explicit TagProbeEDMFilter(const edm::ParameterSet&);
      ~TagProbeEDMFilter();

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
};
#endif
