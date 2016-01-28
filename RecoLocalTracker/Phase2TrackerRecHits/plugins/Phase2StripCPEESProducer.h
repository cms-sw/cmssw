#ifndef RecoLocaltracker_Phase2TrackerRecHits_Phase2StripCPEESProducer_h
#define RecoLocaltracker_Phase2TrackerRecHits_Phase2StripCPEESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/Records/interface/TkStripCPERecord.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/ClusterParameterEstimator.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include <boost/shared_ptr.hpp>
#include <map>

#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPEDummy.h"


class Phase2StripCPEESProducer: public edm::ESProducer {

  public:

    Phase2StripCPEESProducer(const edm::ParameterSet&);
    boost::shared_ptr<ClusterParameterEstimator<Phase2TrackerCluster1D> > produce(const TkStripCPERecord & iRecord);

  private:

    enum CPE_t { DUMMY };
    std::map<std::string, CPE_t> enumMap_;

    CPE_t cpeNum_;
    edm::ParameterSet pset_;
    boost::shared_ptr<ClusterParameterEstimator<Phase2TrackerCluster1D> > cpe_;

};
#endif




