#ifndef DQM_SiStripCommissioningSources_NoiseHVScanTask_h
#define DQM_SiStripCommissioningSources_NoiseHVScanTask_h

#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"
#include <vector>

namespace edm { class ParameterSet; }

/**
   @class NoiseHVScanTask
*/
class NoiseHVScanTask : public CommissioningTask {

  public:

    NoiseHVScanTask(DQMStore *, const FedChannelConnection &, const edm::ParameterSet &);
    virtual ~NoiseHVScanTask();

  private:

    virtual void book();
    virtual void fill( const SiStripEventSummary &, const edm::DetSet<SiStripRawDigi> &);
    virtual void update();
    virtual void fillHVPoint(uint16_t hv);
  
    std::vector<uint32_t> hvDone_;
    int hvCurrent_;

    const uint16_t nstrips, ncmstrips, nbins;
    HistoSet avgnoise_;
    std::vector<unsigned int> vNumOfEntries_;
    std::vector<float> vSumOfContents_;
    std::vector<float> vSumOfSquares_;

};

#endif // DQM_SiStripCommissioningSources_NoiseHVScanTask_h
