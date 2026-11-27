#include <memory>
#include <string>

#include "CondFormats/DataRecord/interface/SiPixelGenErrorDBObjectRcd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFast.h"

template <typename TrackerTraits>
class PixelCPEFastESProducerT : public edm::ESProducer {
public:
  PixelCPEFastESProducerT(const edm::ParameterSet& p);
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
};

using namespace edm;

template <typename TrackerTraits>
PixelCPEFastESProducerT<TrackerTraits>::PixelCPEFastESProducerT(const edm::ParameterSet& p) : pset_(p) {
  auto const& myname = p.getParameter<std::string>("ComponentName");
  auto const& magname = p.getParameter<edm::ESInputTag>("MagneticFieldRecord");
  // the errors and charge bins of all clusters are taken from the GenError
  if (!p.getParameter<bool>("UseErrorsFromTemplates")) {
    throw cms::Exception("Configuration")
        << "PixelCPEFast takes its errors from the GenError payload: UseErrorsFromTemplates must be true";
  }
  useLAWidthFromDB_ = p.getParameter<bool>("useLAWidthFromDB");

  auto cc = setWhatProduced(this, myname);
  magfieldToken_ = cc.consumes(magname);
  pDDToken_ = cc.consumes();
  hTTToken_ = cc.consumes();
  lorentzAngleToken_ = cc.consumes(edm::ESInputTag(""));
  if (useLAWidthFromDB_) {
    lorentzAngleWidthToken_ = cc.consumes(edm::ESInputTag("", "forWidth"));
  }
  genErrorDBObjectToken_ = cc.consumes();
}

template <typename TrackerTraits>
std::unique_ptr<PixelClusterParameterEstimator> PixelCPEFastESProducerT<TrackerTraits>::produce(
    const TkPixelCPERecord& iRecord) {
  // add the new la width object
  const SiPixelLorentzAngle* lorentzAngleWidthProduct = nullptr;
  if (useLAWidthFromDB_) {  // use the width LA
    lorentzAngleWidthProduct = &iRecord.get(lorentzAngleWidthToken_);
  }

  return std::make_unique<PixelCPEFast<TrackerTraits>>(pset_,
                                                       &iRecord.get(magfieldToken_),
                                                       iRecord.get(pDDToken_),
                                                       iRecord.get(hTTToken_),
                                                       &iRecord.get(lorentzAngleToken_),
                                                       &iRecord.get(genErrorDBObjectToken_),
                                                       lorentzAngleWidthProduct);
}

template <typename TrackerTraits>
void PixelCPEFastESProducerT<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  // from PixelCPEBase
  PixelCPEBase::fillPSetDescription(desc);

  // from PixelCPEFast
  PixelCPEFast<TrackerTraits>::fillPSetDescription(desc);

  // used by PixelCPEFast
  desc.add<double>("EdgeClusterErrorX", 50.0);
  desc.add<double>("EdgeClusterErrorY", 85.0);
  desc.add<bool>("UseErrorsFromTemplates", true);
  desc.add<bool>("TruncatePixelCharge", true);

  std::string name = "PixelCPEFast";
  name += TrackerTraits::nameModifier;
  desc.add<std::string>("ComponentName", name);
  desc.add<edm::ESInputTag>("MagneticFieldRecord", edm::ESInputTag());

  descriptions.addWithDefaultLabel(desc);
}

using PixelCPEFastESProducerPhase1 = PixelCPEFastESProducerT<pixelTopology::Phase1>;
DEFINE_FWK_EVENTSETUP_MODULE(PixelCPEFastESProducerPhase1);
using PixelCPEFastESProducerPhase2 = PixelCPEFastESProducerT<pixelTopology::Phase2>;
DEFINE_FWK_EVENTSETUP_MODULE(PixelCPEFastESProducerPhase2);
using PixelCPEFastESProducerHIonPhase1 = PixelCPEFastESProducerT<pixelTopology::HIonPhase1>;
DEFINE_FWK_EVENTSETUP_MODULE(PixelCPEFastESProducerHIonPhase1);
