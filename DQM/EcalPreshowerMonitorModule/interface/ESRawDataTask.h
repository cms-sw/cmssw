#ifndef ESRawDataTask_H
#define ESRawDataTask_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

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
      void beginJob(void);

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

      edm::EDGetTokenT<ESRawDataCollection> dccCollections_;
      edm::EDGetTokenT<FEDRawDataCollection> FEDRawDataCollection_;

      //MonitorElement* meRunNumberErrors_;
      MonitorElement* meL1ADCCErrors_;
      MonitorElement* meBXDCCErrors_;
      MonitorElement* meOrbitNumberDCCErrors_;
      MonitorElement* meL1ADiff_;
      MonitorElement* meBXDiff_;
      MonitorElement* meOrbitNumberDiff_;

      bool init_;
      int runNum_;

};

#endif
