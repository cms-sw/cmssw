#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "RecoLocalTracker/Records/interface/TkStripCPERecord.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/ClusterParameterEstimator.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPE.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPEGeometric.h"

#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"

#include <memory>
#include <map>

class Phase2StripCPEESProducer: public edm::ESProducer {

    public:
    
        Phase2StripCPEESProducer(const edm::ParameterSet&);
        std::shared_ptr<ClusterParameterEstimator<Phase2TrackerCluster1D> > produce(const TkStripCPERecord & iRecord);

    private:

        enum CPE_t { DEFAULT, GEOMETRIC };
        std::map<std::string, CPE_t> enumMap_;

        CPE_t cpeNum_;
        edm::ParameterSet pset_;
        std::shared_ptr<ClusterParameterEstimator<Phase2TrackerCluster1D> > cpe_;

};


Phase2StripCPEESProducer::Phase2StripCPEESProducer(const edm::ParameterSet & p) {
  std::string name = p.getParameter<std::string>("ComponentType");

  enumMap_[std::string("Phase2StripCPE")]          = DEFAULT;
  enumMap_[std::string("Phase2StripCPEGeometric")] = GEOMETRIC;
  if (enumMap_.find(name) == enumMap_.end())
    throw cms::Exception("Unknown StripCPE type") << name;

  cpeNum_ = enumMap_[name];
  pset_ = p.getParameter<edm::ParameterSet>("parameters");
  setWhatProduced(this, name);
}


std::shared_ptr<ClusterParameterEstimator<Phase2TrackerCluster1D> > Phase2StripCPEESProducer::produce(const TkStripCPERecord & iRecord) {

  edm::ESHandle<MagneticField> magfield;
  edm::ESHandle<TrackerGeometry> pDD;

  switch(cpeNum_) {
    case DEFAULT:
      iRecord.getRecord<IdealMagneticFieldRecord>().get(magfield );
      iRecord.getRecord<TrackerDigiGeometryRecord>().get( pDD );
      cpe_ = std::make_shared<Phase2StripCPE>(pset_, *magfield,*pDD);
      break;
    case GEOMETRIC:
      cpe_ = std::make_shared<Phase2StripCPEGeometric>(pset_);
      break;
  }
  return cpe_;

}


#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_MODULE(Phase2StripCPEESProducer);
