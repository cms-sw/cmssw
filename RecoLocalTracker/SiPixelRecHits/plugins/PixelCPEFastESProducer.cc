#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFast.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

// new record
#include "CondFormats/DataRecord/interface/SiPixelGenErrorDBObjectRcd.h"

#include <string>
#include <memory>

class PixelCPEFastESProducer : public edm::ESProducer {
public:
  PixelCPEFastESProducer(const edm::ParameterSet& p);
  std::unique_ptr<PixelClusterParameterEstimator> produce(const TkPixelCPERecord&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magfieldToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> pDDToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> hTTToken_;
  edm::ESGetToken<SiPixelLorentzAngle, SiPixelLorentzAngleRcd> lorentzAngleToken_;
  edm::ESGetToken<SiPixelLorentzAngle, SiPixelLorentzAngleRcd> lorentzAngleWidthToken_;
  edm::ESGetToken<SiPixelGenErrorDBObject, SiPixelGenErrorDBObjectRcd> genErrorDBObjectToken_;

  edm::ParameterSet pset_;
  bool UseErrorsFromTemplates_;
};

using namespace edm;

PixelCPEFastESProducer::PixelCPEFastESProducer(const edm::ParameterSet& p) {
  std::string myname = p.getParameter<std::string>("ComponentName");
  auto magname = p.getParameter<edm::ESInputTag>("MagneticFieldRecord");
  UseErrorsFromTemplates_ = p.getParameter<bool>("UseErrorsFromTemplates");

  pset_ = p;
  auto c = setWhatProduced(this, myname);
  c.setConsumes(magfieldToken_, magname)
      .setConsumes(pDDToken_)
      .setConsumes(hTTToken_)
      .setConsumes(lorentzAngleToken_, edm::ESInputTag(""));
  c.setConsumes(lorentzAngleWidthToken_, edm::ESInputTag("", "forWidth"));
  if (UseErrorsFromTemplates_) {
    c.setConsumes(genErrorDBObjectToken_);
  }
}

std::unique_ptr<PixelClusterParameterEstimator> PixelCPEFastESProducer::produce(const TkPixelCPERecord& iRecord) {
  // add the new la width object
  const SiPixelLorentzAngle* lorentzAngleWidthProduct = nullptr;
  lorentzAngleWidthProduct = &iRecord.get(lorentzAngleWidthToken_);

  const SiPixelGenErrorDBObject* genErrorDBObjectProduct = nullptr;

  // Errors take only from new GenError
  if (UseErrorsFromTemplates_) {  // do only when generrors are needed
    genErrorDBObjectProduct = &iRecord.get(genErrorDBObjectToken_);
    //} else {
    //std::cout<<" pass an empty GenError pointer"<<std::endl;
  }
  return std::make_unique<PixelCPEFast>(pset_,
                                        &iRecord.get(magfieldToken_),
                                        iRecord.get(pDDToken_),
                                        iRecord.get(hTTToken_),
                                        &iRecord.get(lorentzAngleToken_),
                                        genErrorDBObjectProduct,
                                        lorentzAngleWidthProduct);
}

void PixelCPEFastESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // PixelCPEFastESProducer
  edm::ParameterSetDescription desc;
  desc.add<bool>("DoLorentz", false);
  desc.add<double>("lAWidthFPix", 0);
  desc.add<bool>("useLAAlignmentOffsets", false);
  desc.add<bool>("LoadTemplatesFromDB", true);
  desc.add<bool>("UseErrorsFromTemplates", true);
  desc.add<double>("EdgeClusterErrorX", 50.0);
  desc.add<edm::ESInputTag>("MagneticFieldRecord", edm::ESInputTag());
  desc.add<bool>("useLAWidthFromDB", true);
  desc.add<bool>("TruncatePixelCharge", true);
  desc.add<int>("ClusterProbComputationFlag", 0);
  desc.add<double>("lAOffset", 0);
  desc.add<double>("EdgeClusterErrorY", 85.0);
  desc.add<std::string>("ComponentName", "PixelCPEFast");
  desc.add<double>("lAWidthBPix", 0);
  desc.add<bool>("Alpha2Order", true);
  descriptions.add("PixelCPEFastESProducer", desc);
}

DEFINE_FWK_EVENTSETUP_MODULE(PixelCPEFastESProducer);
