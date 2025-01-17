#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include "CondFormats/SiStripObjects/interface/SiStripBackPlaneCorrection.h"
#include "CondFormats/SiStripObjects/interface/SiStripConfObject.h"
#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/EmptyGroupDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripFakeCPE.h"
#include "RecoLocalTracker/Records/interface/TkStripCPERecord.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEfromTrackAngle.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEgeometric.h"

class StripCPEESProducer : public edm::ESProducer {
public:
  StripCPEESProducer(const edm::ParameterSet&);
  std::unique_ptr<StripClusterParameterEstimator> produce(const TkStripCPERecord&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  enum CPE_t { SIMPLE, TRACKANGLE, GEOMETRIC, FAKE };

  CPE_t cpeNum;
  edm::ParameterSet parametersPSet;

  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> pDDToken_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magfieldToken_;
  edm::ESGetToken<SiStripLorentzAngle, SiStripLorentzAngleDepRcd> lorentzAngleToken_;
  edm::ESGetToken<SiStripBackPlaneCorrection, SiStripBackPlaneCorrectionDepRcd> backPlaneCorrectionToken_;
  edm::ESGetToken<SiStripConfObject, SiStripConfObjectRcd> confObjToken_;
  edm::ESGetToken<SiStripLatency, SiStripLatencyRcd> latencyToken_;
};

void StripCPEESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ComponentName", "SimpleStripCPE");

  edm::ParameterSetDescription cpeFromTrackAngleDesc;
  StripCPEfromTrackAngle::fillPSetDescription(cpeFromTrackAngleDesc);

  edm::ParameterSetDescription emptyDesc;

  desc.ifValue(
      edm::ParameterDescription<std::string>("ComponentType", "SimpleStripCPE", true),
      "SimpleStripCPE" >> edm::ParameterDescription<edm::ParameterSetDescription>("parameters", emptyDesc, true) or
          "StripCPEfromTrackAngle" >>
              edm::ParameterDescription<edm::ParameterSetDescription>("parameters", cpeFromTrackAngleDesc, true) or
          "StripCPEgeometric" >>
              edm::ParameterDescription<edm::ParameterSetDescription>("parameters", emptyDesc, true) or
          "FakeStripCPE" >> edm::EmptyGroupDescription());

  descriptions.addDefault(desc);
}

StripCPEESProducer::StripCPEESProducer(const edm::ParameterSet& p) {
  std::string name = p.getParameter<std::string>("ComponentName");
  std::string type = name;
  if (!p.exists("ComponentType"))
    edm::LogWarning("StripCPEESProducer") << " the CPE producer should contain a ComponentType, probably identical to "
                                             "ComponentName in the first step of migration. Falling back to:"
                                          << type;
  else
    type = p.getParameter<std::string>("ComponentType");

  std::map<std::string, CPE_t> enumMap;
  enumMap[std::string("SimpleStripCPE")] = SIMPLE;
  enumMap[std::string("StripCPEfromTrackAngle")] = TRACKANGLE;
  enumMap[std::string("StripCPEgeometric")] = GEOMETRIC;
  enumMap[std::string("FakeStripCPE")] = FAKE;
  if (enumMap.find(type) == enumMap.end())
    throw cms::Exception("Unknown StripCPE type") << type;

  cpeNum = enumMap[type];
  parametersPSet = p.getParameter<edm::ParameterSet>("parameters");
  auto cc = setWhatProduced(this, name);
  pDDToken_ = cc.consumes();
  magfieldToken_ = cc.consumes();
  lorentzAngleToken_ = cc.consumes();
  backPlaneCorrectionToken_ = cc.consumes();
  confObjToken_ = cc.consumes();
  latencyToken_ = cc.consumes();
}

std::unique_ptr<StripClusterParameterEstimator> StripCPEESProducer::produce(const TkStripCPERecord& iRecord) {
  TrackerGeometry const& pDD = iRecord.get(pDDToken_);
  MagneticField const& magfield = iRecord.get(magfieldToken_);
  SiStripLorentzAngle const& lorentzAngle = iRecord.get(lorentzAngleToken_);
  SiStripBackPlaneCorrection const& backPlaneCorrection = iRecord.get(backPlaneCorrectionToken_);
  SiStripConfObject const& confObj = iRecord.get(confObjToken_);
  SiStripLatency const& latency = iRecord.get(latencyToken_);

  std::unique_ptr<StripClusterParameterEstimator> cpe;

  switch (cpeNum) {
    case SIMPLE:
      cpe = std::make_unique<StripCPE>(
          parametersPSet, magfield, pDD, lorentzAngle, backPlaneCorrection, confObj, latency);
      break;

    case TRACKANGLE:
      cpe = std::make_unique<StripCPEfromTrackAngle>(
          parametersPSet, magfield, pDD, lorentzAngle, backPlaneCorrection, confObj, latency);
      break;

    case GEOMETRIC:
      cpe = std::make_unique<StripCPEgeometric>(
          parametersPSet, magfield, pDD, lorentzAngle, backPlaneCorrection, confObj, latency);
      break;

    case FAKE:
      cpe = std::make_unique<StripFakeCPE>();
      break;
  }

  return cpe;
}

DEFINE_FWK_EVENTSETUP_MODULE(StripCPEESProducer);
