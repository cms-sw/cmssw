
#ifndef DQM_SiStripCommissioningClients_PedsFullNoiseHistosUsingDb_H
#define DQM_SiStripCommissioningClients_PedsFullNoiseHistosUsingDb_H
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"

#include "DQM/SiStripCommissioningClients/interface/PedsFullNoiseHistograms.h"

class PedsFullNoiseHistosUsingDb : public CommissioningHistosUsingDb, public PedsFullNoiseHistograms {

  public:

    PedsFullNoiseHistosUsingDb( const edm::ParameterSet & pset,
                                DQMStore*,
                                SiStripConfigDb* const );

    ~PedsFullNoiseHistosUsingDb() override;

    void uploadConfigurations() override;

   private:

    void update( SiStripConfigDb::FedDescriptionsRange );

    void create( SiStripConfigDb::AnalysisDescriptionsV&, Analysis ) override;

    // parameters
    float highThreshold_;
    float lowThreshold_;
    bool disableBadStrips_;
    bool keepStripsDisabled_;
    bool addBadStrips_;

};

#endif // DQM_SiStripCommissioningClients_PedsFullNoiseHistosUsingDb_H

