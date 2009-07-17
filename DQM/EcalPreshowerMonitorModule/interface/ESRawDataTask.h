#ifndef ESRawDataTask_H
#define ESRawDataTask_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DQMStore;

class ESRawDataTask : public edm::EDAnalyzer {

   public:

      ESRawDataTask(const edm::ParameterSet& ps);
      virtual ~ESRawDataTask();

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
      edm::InputTag FEDRawDataCollection_;

      MonitorElement* meRunNumberErrors_;
      MonitorElement* meL1ADCCErrors_;
      MonitorElement* meBXDCCErrors_;
      MonitorElement* meOrbitNumberDCCErrors_;

      bool init_;
      int runNum_;

};

#endif
