#ifndef ESIntegrityTask_H
#define ESIntegrityTask_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DQMStore;

class ESIntegrityTask : public edm::EDAnalyzer {

   public:

      ESIntegrityTask(const edm::ParameterSet& ps);
      virtual ~ESIntegrityTask();

   protected:

      /// Analyze
      void analyze(const edm::Event& e, const edm::EventSetup& c);

      /// BeginJob
      void beginJob(const edm::EventSetup& c);

      /// EndJob
      void endJob(void);

      /// BeginRun
      void beginRun(const edm::Run & r, const edm::EventSetup & c);

      /// EndRun
      void endRun(const edm::Run & r, const edm::EventSetup & c);

      /// Reset
      void reset(void);

      /// Setup
      void setup(void);

      /// Cleanup
      void cleanup(void);

   private:

      int ievt_;

      DQMStore* dqmStore_;

      std::string prefixME_;

      bool enableCleanup_;
      bool mergeRuns_;

      edm::InputTag dccCollections_;
      edm::InputTag kchipCollections_;

      MonitorElement* meGain_;
      MonitorElement* meFED_;
      MonitorElement* meSLinkCRCErr_;
      MonitorElement* meDCCErr_;
      MonitorElement* meDCCCRCErr_;
      MonitorElement* meOptoRX_;
      MonitorElement* meOptoBC_;
      MonitorElement* meFiberStatus_;
      MonitorElement* meKF1_;
      MonitorElement* meKF2_;
      MonitorElement* meKBC_;
      MonitorElement* meKEC_;

      bool init_;
      int runNum_, eCount_, runtype_, seqtype_, dac_, gain_, precision_;
      int firstDAC_, nDAC_, isPed_, vDAC_[5]; 

};

#endif
