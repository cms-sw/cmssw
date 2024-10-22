#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEGeneric.h"
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

class PixelCPEGenericESProducer : public edm::ESProducer {
public:
  PixelCPEGenericESProducer(const edm::ParameterSet& p);
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
  bool useLAWidthFromDB_;
  bool UseErrorsFromTemplates_;
  std::string CPEgenericMode_;  // user's choice of CPE generic
};

using namespace edm;

PixelCPEGenericESProducer::PixelCPEGenericESProducer(const edm::ParameterSet& p) {
  CPEgenericMode_ = p.getParameter<std::string>("ComponentName");
  // Use LA-width from DB. If both (upper and this) are false LA-width is calcuated from LA-offset
  useLAWidthFromDB_ = p.getParameter<bool>("useLAWidthFromDB");
  // Use Alignment LA-offset
  const bool doLorentzFromAlignment = p.getParameter<bool>("doLorentzFromAlignment");
  char const* laLabel = "";  // standard LA, from calibration, label=""
  if (doLorentzFromAlignment) {
    laLabel = "fromAlignment";
  }

  auto magname = p.getParameter<edm::ESInputTag>("MagneticFieldRecord");
  UseErrorsFromTemplates_ = p.getParameter<bool>("UseErrorsFromTemplates");

  pset_ = p;
  auto c = setWhatProduced(this, CPEgenericMode_);
  magfieldToken_ = c.consumes(magname);
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

std::unique_ptr<PixelClusterParameterEstimator> PixelCPEGenericESProducer::produce(const TkPixelCPERecord& iRecord) {
  // add the new la width object
  const SiPixelLorentzAngle* lorentzAngleWidthProduct = nullptr;
  if (useLAWidthFromDB_) {  // use the width LA
    lorentzAngleWidthProduct = &iRecord.get(lorentzAngleWidthToken_);
  }
  //std::cout<<" la width "<<lorentzAngleWidthProduct<<std::endl; //dk

  const SiPixelGenErrorDBObject* genErrorDBObjectProduct = nullptr;

  // Errors take only from new GenError
  if (UseErrorsFromTemplates_) {  // do only when generrors are needed
    genErrorDBObjectProduct = &iRecord.get(genErrorDBObjectToken_);
    //} else {
    //std::cout<<" pass an empty GenError pointer"<<std::endl;
  }

  return std::make_unique<PixelCPEGeneric>(pset_,
                                           &iRecord.get(magfieldToken_),
                                           iRecord.get(pDDToken_),
                                           iRecord.get(hTTToken_),
                                           &iRecord.get(lorentzAngleToken_),
                                           genErrorDBObjectProduct,
                                           lorentzAngleWidthProduct);
}

void PixelCPEGenericESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  // from PixelCPEBase
  PixelCPEBase::fillPSetDescription(desc);

  // from PixelCPEGeneric
  PixelCPEGeneric::fillPSetDescription(desc);

  // specific to PixelCPEGenericESProducer
  desc.add<std::string>("ComponentName", "PixelCPEGeneric");
  desc.add<edm::ESInputTag>("MagneticFieldRecord", edm::ESInputTag(""));
  descriptions.add("_generic_default", desc);
}

DEFINE_FWK_EVENTSETUP_MODULE(PixelCPEGenericESProducer);
