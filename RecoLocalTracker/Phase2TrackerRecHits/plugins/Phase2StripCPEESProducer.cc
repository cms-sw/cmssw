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

class Phase2StripCPEESProducer : public edm::ESProducer {
public:
  Phase2StripCPEESProducer(const edm::ParameterSet&);
  std::unique_ptr<ClusterParameterEstimator<Phase2TrackerCluster1D> > produce(const TkStripCPERecord& iRecord);

private:
  enum CPE_t { DEFAULT, GEOMETRIC };

  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magfieldToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> pDDToken_;
  CPE_t cpeNum_;
  edm::ParameterSet pset_;
};

Phase2StripCPEESProducer::Phase2StripCPEESProducer(const edm::ParameterSet& p) {
  std::string name = p.getParameter<std::string>("ComponentType");

  std::map<std::string, CPE_t> enumMap;
  enumMap[std::string("Phase2StripCPE")] = DEFAULT;
  enumMap[std::string("Phase2StripCPEGeometric")] = GEOMETRIC;
  if (enumMap.find(name) == enumMap.end())
    throw cms::Exception("Unknown StripCPE type") << name;

  cpeNum_ = enumMap[name];
  pset_ = p.getParameter<edm::ParameterSet>("parameters");
  auto c = setWhatProduced(this, name);
  if (cpeNum_ != GEOMETRIC) {
    c.setConsumes(magfieldToken_);
    c.setConsumes(pDDToken_);
  }
}

std::unique_ptr<ClusterParameterEstimator<Phase2TrackerCluster1D> > Phase2StripCPEESProducer::produce(
    const TkStripCPERecord& iRecord) {
  std::unique_ptr<ClusterParameterEstimator<Phase2TrackerCluster1D> > cpe_;
  switch (cpeNum_) {
    case DEFAULT:
      cpe_ = std::make_unique<Phase2StripCPE>(pset_, iRecord.get(magfieldToken_), iRecord.get(pDDToken_));
      break;

    case GEOMETRIC:
      cpe_ = std::make_unique<Phase2StripCPEGeometric>(pset_);
      break;
  }
  return cpe_;
}

#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_MODULE(Phase2StripCPEESProducer);
