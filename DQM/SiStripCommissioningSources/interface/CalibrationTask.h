#ifndef DQM_SiStripCommissioningSources_CalibrationTask_h
#define DQM_SiStripCommissioningSources_CalibrationTask_h

#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"
#include <string>

/**
   @class CalibrationTask
*/
class CalibrationTask : public CommissioningTask {

 public:
  
  CalibrationTask( DQMStore*, const FedChannelConnection&, const sistrip::RunType&, 
                   const char* filename, uint32_t run, const edm::EventSetup& setup );
  ~CalibrationTask() override;
  void setCurrentFolder(const std::string &);
  
 private:

  void book() override;
  void fill( const SiStripEventSummary&,
		     const edm::DetSet<SiStripRawDigi>& ) override;
  void update() override;

  sistrip::RunType runType_;
  
  std::map<std::string,std::vector<HistoSet>> calib1_; // first  APV --> one key for each calChan
  std::map<std::string,std::vector<HistoSet>> calib2_; // second APV --> one key for each calChan

  uint16_t nBins_;
  uint16_t lastCalChan_, lastCalSel_, lastLatency_;
  std::string extrainfo_;
  std::string directory_;
  uint32_t run_;
  std::vector<uint16_t> ped;
};

#endif // DQM_SiStripCommissioningSources_CalibrationTask_h

