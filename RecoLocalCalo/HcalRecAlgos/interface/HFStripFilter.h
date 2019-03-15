#ifndef RecoLocalCalo_HcalRecAlgos_HFStripFilter_h_
#define RecoLocalCalo_HcalRecAlgos_HFStripFilter_h_

#include <memory>

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class HcalChannelQuality;
class HcalSeverityLevelComputer;

class HFStripFilter
{
public:
    // Construct this object with all necessary parameters
    HFStripFilter(double stripThreshold, double maxThreshold,
                  double timeMax, double maxStripTime,
                  double wedgeCut, int gap,
                  int lstrips, int acceptSeverityLevel,
                  int verboseLevel);

    // Destructor
    ~HFStripFilter();

    // The actual rechit tagging is performed by the following function
    void runFilter(HFRecHitCollection& rec,
                   const HcalChannelQuality* myqual,
                   const HcalSeverityLevelComputer* mySeverity) const;

    // Parser function to create this object from a parameter set
    static std::unique_ptr<HFStripFilter> parseParameterSet(
        const edm::ParameterSet& ps);

    // Standard parameter values
    static edm::ParameterSetDescription fillDescription();

private:
    double stripThreshold_;
    double maxThreshold_;
    double timeMax_;
    double maxStripTime_;
    double wedgeCut_;
    int gap_;
    int lstrips_;
    int acceptSeverityLevel_;
    int verboseLevel_;
};

#endif // RecoLocalCalo_HcalRecAlgos_HFStripFilter_h_
