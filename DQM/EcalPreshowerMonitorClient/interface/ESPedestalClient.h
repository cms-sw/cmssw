#ifndef ESPedestalClient_H
#define ESPedestalClient_H

#include "DQM/EcalPreshowerMonitorClient/interface/ESClient.h"

#include <TF1.h>

#include <vector>

//
// class decleration
//

class ESPedestalClient : public ESClient{
   public:
   ESPedestalClient(const edm::ParameterSet&);
   ~ESPedestalClient() override;
   void endJobAnalyze(DQMStore::IGetter&) override;

   private:
   void book(DQMStore::IBooker&) override;

   bool fitPedestal_;

   MonitorElement *hPed_[2][2][40][40];
   MonitorElement *hTotN_[2][2][40][40];

   TF1* fg_;

   std::vector<int> senZ_;
   std::vector<int> senP_;
   std::vector<int> senX_;
   std::vector<int> senY_;

};

#endif  //ESPedestalClient_H
