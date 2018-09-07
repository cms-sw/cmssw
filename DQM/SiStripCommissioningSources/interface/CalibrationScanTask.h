#ifndef DQM_SiStripCommissioningSources_CalibrationScanTask_h
#define DQM_SiStripCommissioningSources_CalibrationScanTask_h

#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"
#include <string>

/**
   @class CalibrationScanTask
*/
class CalibrationScanTask : public CommissioningTask {

 public:
  
  CalibrationScanTask( DQMStore*, const FedChannelConnection&, const sistrip::RunType&, 
                       const char* filename, uint32_t run, const edm::EventSetup& setup );
  ~CalibrationScanTask() override;
  void setCurrentFolder(const std::string &);
  
 private:

  void book() override;
  void fill( const SiStripEventSummary&,const edm::DetSet<SiStripRawDigi>& ) override;
  void update() override;
  
  sistrip::RunType runType_;  
  std::map<std::string,HistoSet> calib1_, calib2_;

  uint16_t nBins_;
  uint32_t lastISHA_,lastVFS_, lastCalChan_, lastCalSel_, lastLatency_;  
  std::vector<uint16_t> ped;
  std::string extrainfo_;
  std::string directory_;
  uint32_t run_;
};

#endif // DQM_SiStripCommissioningSources_CalibrationScanTask_h

