#ifndef L1TAnalyzer_L1TAnalyzer_h
#define L1TAnalyzer_L1TAnalyzer_h

#include <string>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


/* Small class whos main job is to get L1TriggerKey and L1TCSCTPParameters from EventSetup.
 * This way you can check if data was loaded and put there.
 */
class L1TAnalyzer : public edm::EDAnalyzer {
   public:
      explicit L1TAnalyzer(const edm::ParameterSet&);
      ~L1TAnalyzer();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob();
};

#endif
