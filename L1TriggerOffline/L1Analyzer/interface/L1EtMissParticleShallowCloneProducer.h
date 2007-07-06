#ifndef L1Analyzer_L1EtMissParticleShallowCloneProducer_h
#define L1Analyzer_L1EtMissParticleShallowCloneProducer_h
// -*- C++ -*-
//
// Package:    L1Analyzer
// Class:      L1EtMissParticleShallowCloneProducer
// 
/**\class L1EtMissParticleShallowCloneProducer 

 Description: Make shallow clone of L1Extra Etmiss particle

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Alex Tapper
//         Created:  Fri Jan 19 14:30:35 CET 2007
// $Id: L1EtMissParticleShallowCloneProducer.h,v 1.1 2007/02/13 14:49:19 tapper Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class declaration
//

class L1EtMissParticleShallowCloneProducer : public edm::EDProducer {
   public:
      explicit L1EtMissParticleShallowCloneProducer(const edm::ParameterSet&);
      ~L1EtMissParticleShallowCloneProducer();

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&);
      
      edm::InputTag m_l1EtMissSource; // Tag for input EtMiss particle
};

#endif

