#ifndef L1TrigReport_h
#define L1TrigReport_h

/** \class L1TrigReport
 *
 *  
 *  This class is an EDAnalyzer implementing TrigReport (statistics
 *  printed to log file) for L1 triggers
 *
 *  $Date: 2006/10/04 16:02:42 $
 *  $Revision: 1.13 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include<vector>
#include<string>

//
// class declaration
//

class L1TrigReport : public edm::EDAnalyzer {

   public:
      explicit L1TrigReport(const edm::ParameterSet&);
      ~L1TrigReport();
      virtual void endJob();

      virtual void analyze(const edm::Event&, const edm::EventSetup&);

   private:

      edm::InputTag l1ParticleMapTag_;   // Only because decisionWord of L1GTRR is not yet filled!
      edm::InputTag l1GTReadoutRecTag_;  // L1GlobalTriggerReadoutRecord

      unsigned int  nEvents_;            // number of events processed
      unsigned int  nErrors_;            // number of events with error (EDProduct[s] not found)
      unsigned int  nAccepts_;           // number of events accepted by (any) L1 algorithm

      std::vector<unsigned int> l1Accepts_; // number of events accepted by each L1 algorithm
      std::vector<std::string> l1Names_;    // name of each L1 algorithm

      bool init_;                           // vectors initialised or not

};

#endif //L1TrigReport_h
