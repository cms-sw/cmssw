#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPETemplateReco.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "CondFormats/DataRecord/interface/SiPixelGenErrorDBObjectRcd.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/PluginDescription.h"

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "RecoLocalTracker/SiPixelRecHits/plugins/PixelCPENNReco.h"

#include <string>
#include <memory>
#include <array>
#include <filesystem>
#include <vector>

class PixelCPENNRecoESProducer : public edm::ESProducer {
public:
  PixelCPENNRecoESProducer(const edm::ParameterSet& p);
  std::unique_ptr<PixelClusterParameterEstimator> produce(const TkPixelCPERecord&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magfieldToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> pDDToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> hTTToken_;
  edm::ESGetToken<SiPixelLorentzAngle, SiPixelLorentzAngleRcd> lorentzAngleToken_;
  edm::ESGetToken<SiPixelGenErrorDBObject, SiPixelGenErrorDBObjectRcd> genErrorDBObjectToken_;

  std::vector<std::unique_ptr<tensorflow::SessionCache>> modelCachesX_;
  std::vector<std::unique_ptr<tensorflow::SessionCache>> modelCachesY_;

  edm::ParameterSet pset_;
  bool doLorentzFromAlignment_;
  bool useLAFromDB_;
};

using namespace edm;

PixelCPENNRecoESProducer::PixelCPENNRecoESProducer(const edm::ParameterSet& p) {
  std::string myname = p.getParameter<std::string>("ComponentName");

  const auto modelDirectory = p.getParameter<std::string>("modelDirectory");
  const std::array<std::string, 7> xModelNames = {
      "L1U_x_center.keras.pb",
      "L1F_x_center.keras.pb",
      "L2_x_center.keras.pb",
      "L3M_x_center.keras.pb",
      "L3P_x_center.keras.pb",
      "L4M_x_center.keras.pb",
      "L4P_x_center.keras.pb",
  };
  const std::array<std::string, 7> yModelNames = {
      "L1U_y_center.keras.pb",
      "L1F_y_center.keras.pb",
      "L2_y_center.keras.pb",
      "L3M_y_center.keras.pb",
      "L3P_y_center.keras.pb",
      "L4M_y_center.keras.pb",
      "L4P_y_center.keras.pb",
  };

  for (const auto& name : xModelNames) {
    auto path = std::filesystem::path(modelDirectory) / name;
    modelCachesX_.push_back(std::make_unique<tensorflow::SessionCache>(path.string()));
  }
  for (const auto& name : yModelNames) {
    auto path = std::filesystem::path(modelDirectory) / name;
    modelCachesY_.push_back(std::make_unique<tensorflow::SessionCache>(path.string()));
  }

  useLAFromDB_ = p.getParameter<bool>("useLAFromDB");
  doLorentzFromAlignment_ = p.getParameter<bool>("doLorentzFromAlignment");
  pset_ = p;
  auto c = setWhatProduced(this, myname);
  magfieldToken_ = c.consumes();
  pDDToken_ = c.consumes();
  hTTToken_ = c.consumes();
  genErrorDBObjectToken_ = c.consumes();
  if (useLAFromDB_ || doLorentzFromAlignment_) {
    char const* laLabel = doLorentzFromAlignment_ ? "fromAlignment" : "";
    lorentzAngleToken_ = c.consumes(edm::ESInputTag("", laLabel));
  }
}

std::unique_ptr<PixelClusterParameterEstimator> PixelCPENNRecoESProducer::produce(const TkPixelCPERecord& iRecord) {
  std::vector<const tensorflow::Session*> sessionsX;
  std::vector<const tensorflow::Session*> sessionsY;

  sessionsX.reserve(modelCachesX_.size());
  sessionsY.reserve(modelCachesY_.size());

  for (const auto& cache : modelCachesX_)
    sessionsX.push_back(cache->getSession());

  for (const auto& cache : modelCachesY_)
    sessionsY.push_back(cache->getSession());

  const SiPixelLorentzAngle* lorentzAngleProduct = nullptr;
  if (useLAFromDB_ || doLorentzFromAlignment_) {
    lorentzAngleProduct = &iRecord.get(lorentzAngleToken_);
  }
  return std::make_unique<PixelCPENNReco>(pset_,
                                          &iRecord.get(magfieldToken_),
                                          iRecord.get(pDDToken_),
                                          iRecord.get(hTTToken_),
                                          lorentzAngleProduct,
                                          &iRecord.get(genErrorDBObjectToken_),
                                          sessionsX,
                                          sessionsY);
}

void PixelCPENNRecoESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  PixelCPEBase::fillPSetDescription(desc);
  PixelCPENNReco::fillPSetDescription(desc);
  desc.add<std::string>("ComponentName", "PixelCPENNReco");

  desc.add<std::string>("modelDirectory");
  descriptions.add("_NN_default", desc);
}

DEFINE_FWK_EVENTSETUP_MODULE(PixelCPENNRecoESProducer);
