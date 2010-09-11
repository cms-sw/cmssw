#ifndef ESSummaryClient_H
#define ESSummaryClient_H

#include <vector>
#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQM/EcalPreshowerMonitorClient/interface/ESClient.h"

class MonitorElement;
class DQMStore;

class ESSummaryClient : public ESClient {

   public:

      /// Constructor
      ESSummaryClient(const edm::ParameterSet& ps);

      /// Destructor
      virtual ~ESSummaryClient();

      /// Analyze
      void analyze(void);

      /// BeginJob
      void beginJob(DQMStore* dqmStore);

      /// EndJob
      void endJob(void);

      /// BeginRun
      void beginRun(void);

      /// EndRun
      void endRun(void);

      /// Setup
      void setup(void);

      /// Cleanup
      void cleanup(void);

      /// SoftReset
      void softReset(bool flag);

      /// Get Functions
      inline int getEvtPerJob() { return ievt_; }
      inline int getEvtPerRun() { return jevt_; }

      /// Set Clients
      inline void setFriends(std::vector<ESClient*> clients) { clients_ = clients; }

     /// EndLumiAnalyze 
     void endLumiAnalyze();

   private:

      int ievt_;
      int jevt_;

      bool cloneME_;
      bool verbose_;
      bool debug_;
      bool enableCleanup_;

      std::string prefixME_;

      std::vector<ESClient*> clients_;

      DQMStore* dqmStore_;

};

#endif
