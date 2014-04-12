#ifndef DQM_SISTRIPCOMMISSIONINGSOURCES_PEDSFULLNOISETASK_H
#define DQM_SISTRIPCOMMISSIONINGSOURCES_PEDSFULLNOISETASK_H

#include <vector>

#include "DataFormats/Common/interface/DetSet.h"
#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"

// Forward Declarations
namespace edm { class ParameterSet; }
class DQMStore;
class FedChannelConnection;
class SiStripEventSummary;
class SiStripRawDigi;
class TH2S;

/**
   @class PedsFullNoiseTask
*/
class PedsFullNoiseTask : public CommissioningTask {

  public:

    PedsFullNoiseTask(DQMStore * dqm, const FedChannelConnection & conn, const edm::ParameterSet & pset);
    virtual ~PedsFullNoiseTask();

  private:

    virtual void book();
    virtual void fill( const SiStripEventSummary &,
                       const edm::DetSet<SiStripRawDigi> &);
    virtual void update();

    // analysis histograms and related variables
    HistoSet pedhist_, noiseprof_;
    CompactHistoSet noisehist_;
    TH2S * hist2d_;
    std::vector<int16_t> peds_;
    std::vector<float> pedsfl_;
    // keeps track of whether desired number of events were skipped
    bool skipped_;
    // number of events to skip
    uint16_t nskip_;
    // keeps track of whether pedestal step is finished
    bool pedsdone_;
    // number of events to be used for pedestals
    uint16_t nevpeds_;
    // width of the expected noise peak (1 bin/adc hardcoded)
    uint16_t nadcnoise_;
    // number of strips per apv
    uint16_t nstrips_;
    // whether to fill the old-style noise profile
    bool fillnoiseprofile_;
    // for expert debugging only! - whether to use average instead of median CM
    bool useavgcm_;
    // for expert debugging only! - whether to use float pedestals instead of rounded int's
    bool usefloatpeds_;

};

#endif // DQM_SISTRIPCOMMISSIONINGSOURCES_PEDSFULLNOISETASK_H
