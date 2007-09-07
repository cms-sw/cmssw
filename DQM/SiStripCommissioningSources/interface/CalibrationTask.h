#ifndef DQM_SiStripCommissioningSources_CalibrationTask_h
#define DQM_SiStripCommissioningSources_CalibrationTask_h

#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"
#include <string>

/**
   @class CalibrationTask
*/
class CalibrationTask : public CommissioningTask {

 public:
  
  CalibrationTask( DaqMonitorBEInterface*, const FedChannelConnection&, const sistrip::RunType&, 
                   const char* filename, const edm::EventSetup& setup );
  virtual ~CalibrationTask();
  
 private:

  virtual void book();
  virtual void fill( const SiStripEventSummary&,
		     const edm::DetSet<SiStripRawDigi>& );
  virtual void update();
  void checkAndSave(const uint16_t&);
  
  sistrip::RunType runType_;
  
  std::vector<HistoSet> calib_;

  uint16_t nBins_;
  uint16_t lastCalChan_;
  std::string filename_;
  std::vector<uint16_t> ped;

};

#endif // DQM_SiStripCommissioningSources_CalibrationTask_h

