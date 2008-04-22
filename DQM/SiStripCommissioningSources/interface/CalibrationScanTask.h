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
  virtual ~CalibrationScanTask();
  
 private:

  virtual void book();
  virtual void fill( const SiStripEventSummary&,
		     const edm::DetSet<SiStripRawDigi>& );
  virtual void update();
  void checkAndSave(const uint16_t& isha, const uint16_t& vfs );
  
  sistrip::RunType runType_;
  
  HistoSet calib1_, calib2_;

  uint16_t nBins_;
  uint16_t lastISHA_,lastVFS_;
  std::string filename_;
  std::vector<uint16_t> ped;
  uint32_t run_;

};

#endif // DQM_SiStripCommissioningSources_CalibrationScanTask_h

