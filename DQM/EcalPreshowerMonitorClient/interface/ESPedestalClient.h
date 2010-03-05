#ifndef ESPedestalClient_H
#define ESPedestalClient_H


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQM/EcalPreshowerMonitorClient/interface/ESClient.h"
#include "OnlineDB/ESCondDB/interface/ESMonPedestalsDat.h"

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/ESCondDB/interface/ESCondDBInterface.h"
#include "OnlineDB/ESCondDB/interface/ESMonPedestalsDat.h"
#include "DQM/EcalPreshowerMonitorClient/interface/ESPedestalClient.h"

#include <TF1.h>


//
// class decleration
//


class MonitorElement;
class DQMStore;
class ESCondDBInterface;
class RunIOV;
class ESMonRunIOV;

class ESPedestalClient : public ESClient{

   friend class ESSummaryClient;

   public:
   ESPedestalClient(const edm::ParameterSet& ps);
   virtual ~ESPedestalClient();
   void analyze();
   //void beginJob(DQMStore* dqmStore);
   void beginJob(void);
   void endJob(void);
   void beginRun(void);
   void endRun(void);
   void setup(void);
   void cleanup(void);
   /// WriteDB
   void writeDb(ESCondDBInterface* econn, RunIOV* runiov, ESMonRunIOV* moniov, int side);
   
   /// Get Functions
   int  searchxy(int Z, int X, int Y);
   inline int getEvtPerJob() { return ievt_; }
   inline int getEvtPerRun() { return jevt_; }


   private:

   int ievt_;
   int jevt_;
   bool enableCleanup_;
   bool verbose_;
   bool debug_;
   bool fitPedestal_;
   bool PlusSide;
   bool MinusSide;
   int Side_;

   edm::FileInPath lookup_;
   std::string prefixME_;

   DQMStore* dqmStore_;

   MonitorElement *hPed_[2][2][40][40];
   MonitorElement *hTotN_[2][2][40][40];

   TF1 *fg;

   int nLines_;
   //		int runNum_, runtype_, seqtype_, dac_, gain_, precision_;
   //		int firstDAC_, nDAC_, isPed_, vDAC_[5], layer_;

   int senZ_[4288], senP_[4288], senX_[4288], senY_[4288];

};

#endif  //ESPedestalClient_H
