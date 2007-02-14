#ifndef GlobalTriggerAnalyzer_L1GtAnalyzer_h
#define GlobalTriggerAnalyzer_L1GtAnalyzer_h

/**
 * \class L1GtAnalyzer
 * 
 * 
 * 
 * Description: example for L1 Global Trigger analyzer
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// class declaration

class L1GtAnalyzer : public edm::EDAnalyzer {
   public:
      explicit L1GtAnalyzer(const edm::ParameterSet&);
      ~L1GtAnalyzer();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

};

#endif /*GlobalTriggerAnalyzer_L1GtAnalyzer_h*/
