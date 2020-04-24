#ifndef DQM_SISTRIPCOMMISSIONINGSOURCES_PEDSONLYTASK_H
#define DQM_SISTRIPCOMMISSIONINGSOURCES_PEDSONLYTASK_H

#include <vector>

#include "DataFormats/Common/interface/DetSet.h"
#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"

// Forward Declarations
class ApvAnalysisFactory;
class DQMStore;
class FedChannelConnection;
class SiStripEventSummary;
class SiStripRawDigi;

/**
   @class PedsOnlyTask
*/
class PedsOnlyTask : public CommissioningTask 
{
  public:
    PedsOnlyTask( DQMStore *, const FedChannelConnection &);
    ~PedsOnlyTask() override;

  private:
    void book() override;
    void fill( const SiStripEventSummary &,
                       const edm::DetSet<SiStripRawDigi> &) override;
    void update() override;

    std::vector<HistoSet> peds_;

    ApvAnalysisFactory *pApvFactory_; 
};

#endif // DQM_SISTRIPCOMMISSIONINGSOURCES_PEDSONLYTASK_H
