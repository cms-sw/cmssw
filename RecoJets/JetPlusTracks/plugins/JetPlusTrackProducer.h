// -*- C++ -*-
//
// Package:    JetPlusTracks
// Class:      JetPlusTrackProducer
// 
/**\class JetPlusTrackProducer JetPlusTrackProducer.cc JetPlusTrackProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Olga Kodolova,40 R-A12,+41227671273,
//         Created:  Fri Feb 19 10:14:02 CET 2010
// $Id: JetPlusTrackProducer.h,v 1.1 2010/03/04 13:12:36 kodolova Exp $
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
#include "JetPlusTrackCorrector.h"
#include "ZSPJPTJetCorrector.h"

#include <string>

//
// class declaration
//

class JetPlusTrackProducer : public edm::EDProducer {
   public:
      explicit JetPlusTrackProducer(const edm::ParameterSet&);
      ~JetPlusTrackProducer();
      virtual void beginJob();
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob();

   private:
      
// Data      
      JetPlusTrackCorrector* mJPTalgo;
      ZSPJPTJetCorrector*       mZSPalgo; 
      edm::InputTag          src;
      edm::InputTag          srcPVs_;
      std::string            alias;
      bool                   vectorial_;
      bool                   useZSP;
      // ----------member data ---------------------------
};
