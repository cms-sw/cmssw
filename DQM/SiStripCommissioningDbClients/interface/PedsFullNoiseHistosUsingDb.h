// Last commit: $Id: PedsFullNoiseHistosUsingDb.h,v 1.10 2009/06/18 20:52:35 lowette Exp $

#ifndef DQM_SiStripCommissioningClients_PedsFullNoiseHistosUsingDb_H
#define DQM_SiStripCommissioningClients_PedsFullNoiseHistosUsingDb_H
#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"

#include "DQM/SiStripCommissioningClients/interface/PedsFullNoiseHistograms.h"

class PedsFullNoiseHistosUsingDb : public CommissioningHistosUsingDb, public PedsFullNoiseHistograms {

  public:

    PedsFullNoiseHistosUsingDb( const edm::ParameterSet & pset,
                                DQMOldReceiver*,
                                SiStripConfigDb* const );
    PedsFullNoiseHistosUsingDb( const edm::ParameterSet & pset,
                                DQMStore*,
                                SiStripConfigDb* const );
    virtual ~PedsFullNoiseHistosUsingDb();

    virtual void uploadConfigurations();

   private:

    void update( SiStripConfigDb::FedDescriptionsRange );

    void create( SiStripConfigDb::AnalysisDescriptionsV&, Analysis );

    // parameters
    float highThreshold_;
    float lowThreshold_;

};

#endif // DQM_SiStripCommissioningClients_PedsFullNoiseHistosUsingDb_H

