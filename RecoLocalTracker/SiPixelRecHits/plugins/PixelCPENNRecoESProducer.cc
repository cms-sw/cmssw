#include "RecoLocalTracker/SiPixelRecHits/plugins/PixelCPENNReco.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "CondFormats/DataRecord/interface/SiPixelGenErrorDBObjectRcd.h"

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

#include <filesystem>
#include <memory>
#include <string>

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
  edm::ESGetToken<SiPixelLorentzAngle, SiPixelLorentzAngleRcd> lorentzAngleWidthToken_;
  edm::ESGetToken<SiPixelGenErrorDBObject, SiPixelGenErrorDBObjectRcd> genErrorDBObjectToken_;

  std::unique_ptr<cms::Ort::ONNXRuntime> model_;

  edm::ParameterSet pset_;
  bool useLAWidthFromDB_;
  bool UseErrorsFromTemplates_;
};

PixelCPENNRecoESProducer::PixelCPENNRecoESProducer(const edm::ParameterSet& p) {
  const auto modelDirectory = p.getParameter<std::string>("modelDirectory");
  const std::string ModelName = "PixelCPENNReco.onnx";

  auto path = std::filesystem::path(modelDirectory) / ModelName;
  model_ = std::make_unique<cms::Ort::ONNXRuntime>(path.string());

  // Same conditions as PixelCPEGenericESProducer, used for the generic fallback
  useLAWidthFromDB_ = p.getParameter<bool>("useLAWidthFromDB");
  const bool doLorentzFromAlignment = p.getParameter<bool>("doLorentzFromAlignment");
  char const* laLabel = doLorentzFromAlignment ? "fromAlignment" : "";
  UseErrorsFromTemplates_ = p.getParameter<bool>("UseErrorsFromTemplates");

  pset_ = p;
  auto c = setWhatProduced(this, p.getParameter<std::string>("ComponentName"));
  magfieldToken_ = c.consumes(p.getParameter<edm::ESInputTag>("MagneticFieldRecord"));
  pDDToken_ = c.consumes();
  hTTToken_ = c.consumes();
  lorentzAngleToken_ = c.consumes(edm::ESInputTag("", laLabel));
  if (useLAWidthFromDB_) {
    lorentzAngleWidthToken_ = c.consumes(edm::ESInputTag("", "forWidth"));
  }
  if (UseErrorsFromTemplates_) {
    genErrorDBObjectToken_ = c.consumes();
  }
}

std::unique_ptr<PixelClusterParameterEstimator> PixelCPENNRecoESProducer::produce(const TkPixelCPERecord& iRecord) {
  const cms::Ort::ONNXRuntime* model = model_.get();
  const SiPixelLorentzAngle* lorentzAngleWidthProduct = nullptr;
  if (useLAWidthFromDB_) {
    lorentzAngleWidthProduct = &iRecord.get(lorentzAngleWidthToken_);
  }
  const SiPixelGenErrorDBObject* genErrorDBObjectProduct = nullptr;
  if (UseErrorsFromTemplates_) {
    genErrorDBObjectProduct = &iRecord.get(genErrorDBObjectToken_);
  }

  return std::make_unique<PixelCPENNReco>(pset_,
                                          &iRecord.get(magfieldToken_),
                                          iRecord.get(pDDToken_),
                                          iRecord.get(hTTToken_),
                                          &iRecord.get(lorentzAngleToken_),
                                          genErrorDBObjectProduct,
                                          lorentzAngleWidthProduct,
                                          model);
}

void PixelCPENNRecoESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  // from PixelCPEBase
  PixelCPEBase::fillPSetDescription(desc);

  // from PixelCPENNReco (includes PixelCPEGeneric)
  PixelCPENNReco::fillPSetDescription(desc);

  // specific to PixelCPENNRecoESProducer
  desc.add<std::string>("ComponentName", "PixelCPENNReco");
  desc.add<edm::ESInputTag>("MagneticFieldRecord", edm::ESInputTag(""));
  desc.add<std::string>("modelDirectory");
  descriptions.add("_NN_default", desc);
}

DEFINE_FWK_EVENTSETUP_MODULE(PixelCPENNRecoESProducer);
