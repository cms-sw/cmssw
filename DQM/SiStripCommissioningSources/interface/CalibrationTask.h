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
  
 private:

  void book() override;
  void fill( const SiStripEventSummary&,
		     const edm::DetSet<SiStripRawDigi>& ) override;
  void update() override;
  void checkAndSave(const uint16_t&);
  void directory( std::stringstream&,
                  uint32_t run_number = 0 );

  sistrip::RunType runType_;
  
  std::vector<HistoSet> calib_;

  uint16_t nBins_;
  uint16_t lastCalChan_;
  std::string filename_;
  std::vector<uint16_t> ped;
  uint32_t run_;
  MonitorElement* calchanElement_;
};

#endif // DQM_SiStripCommissioningSources_CalibrationTask_h

