#ifndef DQM_SiStripCommissioningSources_FedCablingTask_h
#define DQM_SiStripCommissioningSources_FedCablingTask_h

#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"
#include <vector>

/**
   @class FedCablingTask

   

   When this CommissioningTask object is created by the
   CommissioningSource "steering , it is stored in map using FEC key created
   using the 32-bit device id found within the "trigger FED"
   buffer. (This is different to the method of using the FED key to
   identify the CommissioningTask object


*/
class FedCablingTask : public CommissioningTask {

 public:
  
  FedCablingTask( DaqMonitorBEInterface*, const FedChannelConnection& );
  virtual ~FedCablingTask();
  
 private:
  
  
  virtual void book();
  virtual void fill( const SiStripEventSummary&, 
		     const uint16_t& fed_id,
		     const std::map<uint16_t,float>& fed_ch );
  virtual void update();
  
  /** HistoSet for FED cabling. First element contains histo info for
      FED id, second element contains histo info for FED channel. */
  vector<HistoSet> cabling_;
  
};

#endif // DQM_SiStripCommissioningSources_FedCablingTask_h

