#ifndef DQM_SiStripCommissioningSources_OptoScanTask_H
#define DQM_SiStripCommissioningSources_OptoScanTask_H

#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"
#include <vector>

/**
   @class OptoScanTask
*/
class OptoScanTask : public CommissioningTask {

  public:

    OptoScanTask( DQMStore * dqm, const FedChannelConnection & conn );
    ~OptoScanTask() override;

  private:

    void book() override;
    void fill( const SiStripEventSummary & summary,
                       const edm::DetSet<SiStripRawDigi> & digis ) override;
    void update() override;
    void locateTicks( const edm::DetSet<SiStripRawDigi> & scope_mode_data,
                      std::pair<float,float> & digital_range, 
                      std::vector<float> & baseline,
                      float & baseline_rms );

    std::vector< std::vector<HistoSet> > opto_;
    uint16_t nBins_;

};

#endif // DQM_SiStripCommissioningSources_OptoScanTask_H

