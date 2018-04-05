#ifndef ESSummaryClient_H
#define ESSummaryClient_H

#include "DQM/EcalPreshowerMonitorClient/interface/ESClient.h"

class ESSummaryClient : public ESClient {

   public:

      /// Constructor
      ESSummaryClient(const edm::ParameterSet& ps);

      /// Destructor
      ~ESSummaryClient() override;

      /// Analyze
      void endLumiAnalyze(DQMStore::IGetter&) override;
      void endJobAnalyze(DQMStore::IGetter&) override;

 private:
      void book(DQMStore::IBooker&) override;

      void fillReportSummary(DQMStore::IGetter&);

      MonitorElement* meReportSummary_;
      MonitorElement* meReportSummaryContents_[2][2];
      MonitorElement* meReportSummaryMap_;
};

#endif
