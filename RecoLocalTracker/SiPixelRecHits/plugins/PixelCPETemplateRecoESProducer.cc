#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPETemplateReco.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <string>
#include <memory>

class PixelCPETemplateRecoESProducer : public edm::ESProducer {
public:
  PixelCPETemplateRecoESProducer(const edm::ParameterSet& p);
  std::unique_ptr<PixelClusterParameterEstimator> produce(const TkPixelCPERecord&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magfieldToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> pDDToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> hTTToken_;
  edm::ESGetToken<SiPixelLorentzAngle, SiPixelLorentzAngleRcd> lorentzAngleToken_;
  edm::ESGetToken<SiPixelTemplateDBObject, SiPixelTemplateDBObjectESProducerRcd> templateDBobjectToken_;

  edm::ParameterSet pset_;
  bool DoLorentz_;
};

using namespace edm;

PixelCPETemplateRecoESProducer::PixelCPETemplateRecoESProducer(const edm::ParameterSet& p) {
  std::string myname = p.getParameter<std::string>("ComponentName");

  //DoLorentz_ = p.getParameter<bool>("DoLorentz"); // True when LA from alignment is used
  DoLorentz_ = p.getParameter<bool>("DoLorentz");

  pset_ = p;
  auto c = setWhatProduced(this, myname);
  c.setConsumes(magfieldToken_).setConsumes(pDDToken_).setConsumes(hTTToken_).setConsumes(templateDBobjectToken_);
  if (DoLorentz_) {
    c.setConsumes(lorentzAngleToken_, edm::ESInputTag("", "fromAlignment"));
  }
  //std::cout<<" from ES Producer Templates "<<myname<<" "<<DoLorentz_<<std::endl;  //dk
}

std::unique_ptr<PixelClusterParameterEstimator> PixelCPETemplateRecoESProducer::produce(
    const TkPixelCPERecord& iRecord) {
  // Normal, deafult LA actually is NOT needed
  // null is ok becuse LA is not use by templates in this mode
  const SiPixelLorentzAngle* lorentzAngleProduct = nullptr;
  if (DoLorentz_) {  //  LA correction from alignment
    lorentzAngleProduct = &iRecord.get(lorentzAngleToken_);
  }

  return std::make_unique<PixelCPETemplateReco>(pset_,
                                                &iRecord.get(magfieldToken_),
                                                iRecord.get(pDDToken_),
                                                iRecord.get(hTTToken_),
                                                lorentzAngleProduct,
                                                &iRecord.get(templateDBobjectToken_));
}

void PixelCPETemplateRecoESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  // from PixelCPEBase
  PixelCPEBase::fillPSetDescription(desc);

  // from PixelCPETemplateReco
  PixelCPETemplateReco::fillPSetDescription(desc);

  // specific to PixelCPETemplateRecoESProducer
  desc.add<std::string>("ComponentName", "PixelCPETemplateReco");
  desc.add<bool>("DoLorentz", true);
  descriptions.add("_templates_default", desc);
}

DEFINE_FWK_EVENTSETUP_MODULE(PixelCPETemplateRecoESProducer);
