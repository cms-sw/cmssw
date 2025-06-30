#include <memory>
#include <string>
#include <alpaka/alpaka.hpp>
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/alpaka/PixelCPEFastParamsCollection.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFastParamsHost.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"

#include "CondFormats/DataRecord/interface/SiPixelGenErrorDBObjectRcd.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  template <typename TrackerTraits>
  class PixelCPEFastParamsESProducerAlpaka : public ESProducer {
  public:
    PixelCPEFastParamsESProducerAlpaka(edm::ParameterSet const& iConfig);
    std::unique_ptr<PixelCPEFastParamsHost<TrackerTraits>> produce(const TkPixelCPERecord& iRecord);

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

  template <typename TrackerTraits>
  PixelCPEFastParamsESProducerAlpaka<TrackerTraits>::PixelCPEFastParamsESProducerAlpaka(const edm::ParameterSet& p)
      : ESProducer(p), pset_(p) {
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

  template <typename TrackerTraits>
  std::unique_ptr<PixelCPEFastParamsHost<TrackerTraits>> PixelCPEFastParamsESProducerAlpaka<TrackerTraits>::produce(
      const TkPixelCPERecord& iRecord) {
    // add the new la width object
    const SiPixelLorentzAngle* lorentzAngleWidthProduct = &iRecord.get(lorentzAngleWidthToken_);

    const SiPixelGenErrorDBObject* genErrorDBObjectProduct = nullptr;

    // Errors take only from new GenError
    if (useErrorsFromTemplates_) {  // do only when generrors are needed
      genErrorDBObjectProduct = &iRecord.get(genErrorDBObjectToken_);
      //} else {
      //std::cout<<" pass an empty GenError pointer"<<std::endl;
    }
    return std::make_unique<PixelCPEFastParamsHost<TrackerTraits>>(pset_,
                                                                   &iRecord.get(magfieldToken_),
                                                                   iRecord.get(pDDToken_),
                                                                   iRecord.get(hTTToken_),
                                                                   &iRecord.get(lorentzAngleToken_),
                                                                   genErrorDBObjectProduct,
                                                                   lorentzAngleWidthProduct);
  }

  template <typename TrackerTraits>
  void PixelCPEFastParamsESProducerAlpaka<TrackerTraits>::fillDescriptions(
      edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    // from PixelCPEBase
    PixelCPEBase::fillPSetDescription(desc);

    // from PixelCPEFast
    PixelCPEFastParamsHost<TrackerTraits>::fillPSetDescription(desc);

    // used by PixelCPEFast
    desc.add<double>("EdgeClusterErrorX", 50.0);
    desc.add<double>("EdgeClusterErrorY", 85.0);
    desc.add<bool>("UseErrorsFromTemplates", true);
    desc.add<bool>("TruncatePixelCharge", true);

    std::string name = "PixelCPEFastParams";
    name += TrackerTraits::nameModifier;
    desc.add<std::string>("ComponentName", name);
    desc.add<edm::ESInputTag>("MagneticFieldRecord", edm::ESInputTag());

    descriptions.addWithDefaultLabel(desc);
  }

  using PixelCPEFastParamsESProducerAlpakaPhase1 = PixelCPEFastParamsESProducerAlpaka<pixelTopology::Phase1>;
  using PixelCPEFastParamsESProducerAlpakaHIonPhase1 = PixelCPEFastParamsESProducerAlpaka<pixelTopology::HIonPhase1>;
  using PixelCPEFastParamsESProducerAlpakaPhase2 = PixelCPEFastParamsESProducerAlpaka<pixelTopology::Phase2>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(PixelCPEFastParamsESProducerAlpakaPhase1);
DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(PixelCPEFastParamsESProducerAlpakaHIonPhase1);
DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(PixelCPEFastParamsESProducerAlpakaPhase2);
