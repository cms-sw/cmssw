#ifndef FastSimulation_MuonSimHitProducer_MuonSimHitProducer_h
#define FastSimulation_MuonSimHitProducer_MuonSimHitProducer_h

//
// Package:    MuonSimHitProducer
// Class:      MuonSimHitProducer
// 
/**\class MuonSimHitProducer MuonSimHitProducer.cc FastSimulation/MuonSimHitProducer/src/MuonSimHitProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
//  Author:  Martijn Mulders
// Created:  Wed July 11 12:37:24 CET 2007
// $Id: MuonSimHitProducer.h,v 1.0 2007/07/11 13:53:50 mulders Exp $
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/EDProducer.h"

// FastSimulation headers
class SimpleL1MuGMTCand;
class FML1EfficiencyHandler;
class FML1PtSmearer;
class FML3EfficiencyHandler; 
class FML3PtSmearer;
class FMGLfromL3EfficiencyHandler; 
class FMGLfromL3TKEfficiencyHandler; 
class FMGLfromTKEfficiencyHandler; 

class RandomEngine;

namespace reco { 
  class Muon;
}

namespace edm { 
  class ParameterSet;
  class Event;
  class EventSetup;
}

// Data Formats
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

//
// class declaration
//

class MuonSimHitProducer : public edm::EDProducer {
   public:

      explicit MuonSimHitProducer(const edm::ParameterSet&);
      ~MuonSimHitProducer();

   private:

      const RandomEngine * random;

      typedef std::vector<SimpleL1MuGMTCand*> FML1Muons;
      typedef std::vector<L1MuGMTCand> L1MuonsContainer;

      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      void readParameters(const edm::ParameterSet&, const edm::ParameterSet& );
      void reconstruct();
      void loadL1Muons(L1MuonsContainer & c) const;
      void loadL3Muons(reco::MuonCollection & c) const;
      void loadGLMuons(reco::MuonCollection & c) const;
    
  // ---------- member data ---------------------------

      FML1Muons  mySimpleL1MuonCands;
      FML1EfficiencyHandler * myL1EfficiencyHandler;
      FML1PtSmearer * myL1PtSmearer;

      reco::MuonCollection  mySimpleL3MuonCands;
      FML3EfficiencyHandler * myL3EfficiencyHandler;
      FML3PtSmearer * myL3PtSmearer;

      reco::MuonCollection  mySimpleGLMuonCands;
      FMGLfromL3EfficiencyHandler * myGLfromL3EfficiencyHandler;
      FMGLfromL3TKEfficiencyHandler * myGLfromL3TKEfficiencyHandler;
      FMGLfromTKEfficiencyHandler * myGLfromTKEfficiencyHandler;
      FML3PtSmearer * myGLPtSmearer;
      
  // ----------- parameters ---------------------------- 
      bool debug_;
      bool fullPattern_;
      bool doL1_ , doL3_ , doGL_;
      std::string theSimModuleLabel_ , theSimModuleProcess_, theTrkModuleLabel_ ;
      double minEta_ ,  maxEta_;
  // ----------- counters ------------------------------
      int   nMuonTot , nL1MuonTot , nL3MuonTot , nGLMuonTot;
};

#endif
