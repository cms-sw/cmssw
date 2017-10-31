#ifndef DQM_SiStripCommissioningSources_FedCablingTask_h
#define DQM_SiStripCommissioningSources_FedCablingTask_h

#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"
#include <vector>

/**
   @class FedCablingTask

   This object is stored in the TaskMap using FecKey as the key,
   rather than FedKey as for the other commissioning tasks.
*/
class FedCablingTask : public CommissioningTask {

 public:
  
  FedCablingTask( DQMStore*, const FedChannelConnection& );
  ~FedCablingTask() override;
  
 private:
  
  void book() override;
  void fill( const SiStripEventSummary&, 
		     const uint16_t& fed_id,
		     const std::map<uint16_t,float>& fed_ch ) override;
  void update() override;
  
  /** HistoSet for FED cabling. First element contains histo info for
      FED id, second element contains histo info for FED channel. */
  std::vector<HistoSet> histos_;
  
};

#endif // DQM_SiStripCommissioningSources_FedCablingTask_h

