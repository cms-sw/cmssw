
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
    float highThreshold_; // higher threshold for the zero suppression
    float lowThreshold_;  // lower threshold for the zero suppression
    bool  disableBadStrips_; // to disable bad strips flagged by the analysis in the upload
    bool  keepStripsDisabled_; // keep bad strips from previous runs as bad
    bool  skipEmptyStrips_;  // skip empty strips i.e. don't flag as bad
    bool  uploadOnlyStripBadChannelBit_;

    // Perform a selective upload either for or excluding a certain set of FEDs                                                                                                                
    bool allowSelectiveUpload_;

    ///////
    bool uploadPedsFullNoiseDBTable_;
};

#endif // DQM_SiStripCommissioningClients_PedsFullNoiseHistosUsingDb_H

