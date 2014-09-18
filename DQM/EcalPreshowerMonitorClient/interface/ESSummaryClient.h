#ifndef ESSummaryClient_H
#define ESSummaryClient_H

#include "DQM/EcalPreshowerMonitorClient/interface/ESClient.h"

class ESSummaryClient : public ESClient {

   public:

      /// Constructor
      ESSummaryClient(const edm::ParameterSet& ps);

      /// Destructor
      virtual ~ESSummaryClient();

      /// Analyze
      void endLumiAnalyze(DQMStore::IGetter&) override;

 private:
      void book(DQMStore::IBooker&) override;
};

#endif
