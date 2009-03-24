#ifndef PhysicsTools_TagAndProbe_TagProbeProducer_h
#define PhysicsTools_TagAndProbe_TagProbeProducer_h
// -*- C++ -*-
//
// Package:     TagAndProbe
// Class  :     TagProbeProducer
// 
/**\class TagProbeProducer TagProbeProducer.h PhysicsTools/TagAndProbe/interface/TagProbeProducer.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Wed Apr 16 10:08:13 CDT 2008
// $Id: TagProbeProducer.h,v 1.2 2008/07/30 13:38:24 srappocc Exp $
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


// forward declarations

class TagProbeProducer : public edm::EDProducer 
{
   public:
      explicit TagProbeProducer(const edm::ParameterSet&);
      ~TagProbeProducer();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
      
      edm::InputTag tagCollection_;
      edm::InputTag probeCollection_;

      double massMinCut_;
      double massMaxCut_;
      double delRMinCut_;
      double delRMaxCut_;

      bool requireOS_;
};

#endif
