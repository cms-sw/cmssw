#ifndef DQM_SISTRIPCOMMISSIONINGSOURCES_PEDSFULLNOISETASK_H
#define DQM_SISTRIPCOMMISSIONINGSOURCES_PEDSFULLNOISETASK_H

#include <vector>

#include "DataFormats/Common/interface/DetSet.h"
#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"

// Forward Declarations
class DQMStore;
class FedChannelConnection;
class SiStripEventSummary;
class SiStripRawDigi;

/**
   @class PedsFullNoiseTask
*/
class PedsFullNoiseTask : public CommissioningTask {

  public:

    PedsFullNoiseTask( DQMStore * dqm, const FedChannelConnection & conn);
    virtual ~PedsFullNoiseTask();

  private:

    virtual void book();
    virtual void fill( const SiStripEventSummary &,
                       const edm::DetSet<SiStripRawDigi> &);
    virtual void update();

    // analysis histograms
    HistoSet pedhist_, pedroughhist_;
    std::vector<HistoSet> cmhist_;
    CompactHistoSet noisehist_;
    // keeps track of the mean of the noise histograms from value instead of histogram bins
    std::vector<float> noiseSum_, noiseNum_;
    // keeps track of whether desired number of events were skipped
    bool skipped_;
    // number of events to skip
    uint16_t nskip_;
    // keeps track of whether "stable temperature" has been reached
    bool tempstable_;
    // number of events before assumed temperature stabilization
    uint16_t ntempstab_;
    // width of the expected noise peak (1 bin/adc hardcoded)
    uint16_t nadcnoise_;
    // number of strips per apv
    uint16_t nstrips_;

};

#endif // DQM_SISTRIPCOMMISSIONINGSOURCES_PEDSFULLNOISETASK_H
