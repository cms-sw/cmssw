#ifndef HLTrigReport_h
#define HLTrigReport_h

/** \class HLTrigReport
 *
 *  
 *  This class is an EDAnalyzer implementing TrigReport (statistics
 *  printed to log file) for HL triggers
 *
 *  $Date: 2010/05/07 15:42:50 $
 *  $Revision: 1.5 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include<vector>
#include<string>

//
// class declaration
//

class HLTrigReport : public edm::EDAnalyzer {

   public:
      explicit HLTrigReport(const edm::ParameterSet&);
      ~HLTrigReport();

      virtual void beginRun(edm::Run const &, edm::EventSetup const&);

      virtual void endJob();

      virtual void analyze(const edm::Event&, const edm::EventSetup&);

      void dumpReport();

   private:

      edm::InputTag hlTriggerResults_;  // Input tag for TriggerResults

      unsigned int  nEvents_;           // number of events processed

      unsigned int  nWasRun_;           // # where at least one HLT was run
      unsigned int  nAccept_;           // # of accepted events
      unsigned int  nErrors_;           // # where at least one HLT had error

      std::vector<unsigned int> hlWasRun_; // # where HLT[i] was run
      std::vector<unsigned int> hltL1s_;   // # of events after L1 seed
      std::vector<unsigned int> hltPre_;   // # of events after HLT prescale
      std::vector<unsigned int> hlAccept_; // # of events accepted by HLT[i]
      std::vector<unsigned int> hlErrors_; // # of events with error in HLT[i]

      std::vector<int> posL1s_;            // pos # of last L1 seed
      std::vector<int> posPre_;            // pos # of last HLT prescale
      std::vector<std::string>  hlNames_;  // name of each HLT algorithm

      HLTConfigProvider hltConfig_;        // to get configuration for L1s/Pre

};

#endif //HLTrigReport_h
