#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CondFormats/DataRecord/interface/SiPhase2OuterTrackerLorentzAngleRcd.h"

#include "RecoLocalTracker/Records/interface/TkPhase2OTCPERecord.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/ClusterParameterEstimator.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPE.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPEGeometric.h"

#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"

#include <memory>
#include <map>

class Phase2StripCPEESProducer : public edm::ESProducer {
public:
  Phase2StripCPEESProducer(const edm::ParameterSet&);
  std::unique_ptr<ClusterParameterEstimator<Phase2TrackerCluster1D> > produce(const TkPhase2OTCPERecord& iRecord);

private:
  enum CPE_t { DEFAULT, GEOMETRIC };

  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magfieldToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> pDDToken_;
  edm::ESGetToken<SiPhase2OuterTrackerLorentzAngle, SiPhase2OuterTrackerLorentzAngleRcd> lorentzAngleToken_;
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
    magfieldToken_ = c.consumes();
    pDDToken_ = c.consumes();
    lorentzAngleToken_ = c.consumes();
  }
}

std::unique_ptr<ClusterParameterEstimator<Phase2TrackerCluster1D> > Phase2StripCPEESProducer::produce(
    const TkPhase2OTCPERecord& iRecord) {
  std::unique_ptr<ClusterParameterEstimator<Phase2TrackerCluster1D> > cpe_;
  switch (cpeNum_) {
    case DEFAULT:
      cpe_ = std::make_unique<Phase2StripCPE>(
          pset_, iRecord.get(magfieldToken_), iRecord.get(pDDToken_), iRecord.get(lorentzAngleToken_));
      break;

    case GEOMETRIC:
      cpe_ = std::make_unique<Phase2StripCPEGeometric>(pset_);
      break;
  }
  return cpe_;
}

#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_MODULE(Phase2StripCPEESProducer);
