#ifndef DQM_SiStripCommissioningSources_OptoBiasAndGainScanTask_h
#define DQM_SiStripCommissioningSources_OptoBiasAndGainScanTask_h

#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"
#include <vector>

/**
   @class OptoBiasAndGainScanTask
*/
class OptoBiasAndGainScanTask : public CommissioningTask {

 public:
  
  OptoBiasAndGainScanTask( DaqMonitorBEInterface*, const FedChannelConnection& );
  virtual ~OptoBiasAndGainScanTask();
  
 private: // ----- private methods -----

  virtual void book( const FedChannelConnection& );
  virtual void fill( const SiStripEventSummary&,
		     const edm::DetSet<SiStripRawDigi>& );
  virtual void update();

  void locateTicks( const edm::DetSet<SiStripRawDigi>& scope_mode_data,
		    pair< uint16_t, uint16_t >& digital_range, 
		    bool first_tick_only = false );
  
 private: // ----- data members -----

  /** "Histo sets" for the various gain settings and digital 0 and 1
      levels. First index is gain (0->3) and the second index is the
      digital level ("0" or "1"). */
  vector< vector<HistoSet> > opto_;

  uint16_t nBins_;

};

#endif // DQM_SiStripCommissioningSources_OptoBiasAndGainScanTask_h

