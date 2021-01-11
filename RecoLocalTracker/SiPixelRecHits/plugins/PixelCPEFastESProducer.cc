#include <memory>
#include <string>

#include "CondFormats/DataRecord/interface/SiPixelGenErrorDBObjectRcd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFast.h"

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
  bool useErrorsFromTemplates_;
};

using namespace edm;

PixelCPEFastESProducer::PixelCPEFastESProducer(const edm::ParameterSet& p) : pset_(p) {
  auto const& myname = p.getParameter<std::string>("ComponentName");
  auto const& magname = p.getParameter<edm::ESInputTag>("MagneticFieldRecord");
  useErrorsFromTemplates_ = p.getParameter<bool>("UseErrorsFromTemplates");

  auto cc = setWhatProduced(this, myname);
  magfieldToken_ = cc.consumes(magname);
  pDDToken_ = cc.consumes();
  hTTToken_ = cc.consumes();
  lorentzAngleToken_ = cc.consumes(edm::ESInputTag(""));
  lorentzAngleWidthToken_ = cc.consumes(edm::ESInputTag("", "forWidth"));
  if (useErrorsFromTemplates_) {
    genErrorDBObjectToken_ = cc.consumes();
  }
}

std::unique_ptr<PixelClusterParameterEstimator> PixelCPEFastESProducer::produce(const TkPixelCPERecord& iRecord) {
  // add the new la width object
  const SiPixelLorentzAngle* lorentzAngleWidthProduct = nullptr;
  lorentzAngleWidthProduct = &iRecord.get(lorentzAngleWidthToken_);

  const SiPixelGenErrorDBObject* genErrorDBObjectProduct = nullptr;

  // Errors take only from new GenError
  if (useErrorsFromTemplates_) {  // do only when generrors are needed
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
  edm::ParameterSetDescription desc;

  // from PixelCPEBase
  PixelCPEBase::fillPSetDescription(desc);

  // used by PixelCPEFast
  desc.add<double>("EdgeClusterErrorX", 50.0);
  desc.add<double>("EdgeClusterErrorY", 85.0);
  desc.add<bool>("UseErrorsFromTemplates", true);
  desc.add<bool>("TruncatePixelCharge", true);

  // specific to PixelCPEFastESProducer
  desc.add<std::string>("ComponentName", "PixelCPEFast");
  desc.add<edm::ESInputTag>("MagneticFieldRecord", edm::ESInputTag());
  desc.add<bool>("useLAAlignmentOffsets", false);
  desc.add<bool>("DoLorentz", false);

  descriptions.add("PixelCPEFastESProducer", desc);
}

DEFINE_FWK_EVENTSETUP_MODULE(PixelCPEFastESProducer);
