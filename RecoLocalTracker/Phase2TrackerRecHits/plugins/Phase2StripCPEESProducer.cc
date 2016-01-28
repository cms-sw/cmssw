
#include "RecoLocalTracker/Phase2TrackerRecHits/plugins/Phase2StripCPEESProducer.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPEDummy.h"


Phase2StripCPEESProducer::Phase2StripCPEESProducer(const edm::ParameterSet & p) {
  std::string name = p.getParameter<std::string>("ComponentType");

  enumMap_[std::string("Phase2StripCPEDummy")] = DUMMY;
  if (enumMap_.find(name) == enumMap_.end())
    throw cms::Exception("Unknown StripCPE type") << name;

  cpeNum_ = enumMap_[name];
  pset_ = p;
  setWhatProduced(this, name);
}


boost::shared_ptr<ClusterParameterEstimator<Phase2TrackerCluster1D> > Phase2StripCPEESProducer::produce(const TkStripCPERecord & iRecord) {

  switch(cpeNum_) {

    case DUMMY:
      cpe_ = boost::shared_ptr<ClusterParameterEstimator<Phase2TrackerCluster1D> >(new Phase2StripCPEDummy());
      break;

  }

  return cpe_;
}


#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_MODULE(Phase2StripCPEESProducer);
