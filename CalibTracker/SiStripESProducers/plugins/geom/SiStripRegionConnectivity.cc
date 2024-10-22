#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripRegionCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripRegionCabling.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

using namespace sistrip;

class SiStripRegionConnectivity : public edm::ESProducer {
public:
  SiStripRegionConnectivity(const edm::ParameterSet&);
  ~SiStripRegionConnectivity() override;

  std::unique_ptr<SiStripRegionCabling> produceRegionCabling(const SiStripRegionCablingRcd&);

private:
  edm::ESGetToken<SiStripDetCabling, SiStripDetCablingRcd> detcablingToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkgeomToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;

  /** Number of regions in eta,phi */
  uint32_t etadivisions_;
  uint32_t phidivisions_;

  /** Tracker extent in eta */
  double etamax_;
};

SiStripRegionConnectivity::SiStripRegionConnectivity(const edm::ParameterSet& pset)
    : etadivisions_(pset.getUntrackedParameter<unsigned int>("EtaDivisions", 10)),
      phidivisions_(pset.getUntrackedParameter<unsigned int>("PhiDivisions", 10)),
      etamax_(pset.getUntrackedParameter<double>("EtaMax", 2.4))

{
  auto cc = setWhatProduced(this, &SiStripRegionConnectivity::produceRegionCabling);
  detcablingToken_ = cc.consumes();
  tkgeomToken_ = cc.consumes();
  tTopoToken_ = cc.consumes();
}

SiStripRegionConnectivity::~SiStripRegionConnectivity() {}

std::unique_ptr<SiStripRegionCabling> SiStripRegionConnectivity::produceRegionCabling(
    const SiStripRegionCablingRcd& iRecord) {
  const auto& detcabling = iRecord.get(detcablingToken_);
  const auto& tkgeom = iRecord.get(tkgeomToken_);
  const auto& tTopo = iRecord.get(tTopoToken_);

  //here build an object of type SiStripRegionCabling using the information from class SiStripDetCabling **PLUS** the geometry.

  //Construct region cabling object
  auto RegionConnections = std::make_unique<SiStripRegionCabling>(etadivisions_, phidivisions_, etamax_);

  //Construct region cabling map
  SiStripRegionCabling::Cabling regioncabling(
      etadivisions_ * phidivisions_,
      SiStripRegionCabling::RegionCabling(
          SiStripRegionCabling::ALLSUBDETS,
          SiStripRegionCabling::WedgeCabling(SiStripRegionCabling::ALLLAYERS, SiStripRegionCabling::ElementCabling())));

  //Loop det cabling
  for (const auto& idet : detcabling.getDetCabling()) {
    if (!idet.first || (idet.first == sistrip::invalid32_))
      continue;

    // Check if geom det unit exists
    auto geom_det = tkgeom.idToDetUnit(DetId(idet.first));
    auto strip_det = dynamic_cast<StripGeomDetUnit const*>(geom_det);
    if (!strip_det) {
      continue;
    }

    //Calculate region from geometry
    double eta = tkgeom.idToDet(DetId(idet.first))->position().eta();
    double phi = tkgeom.idToDet(DetId(idet.first))->position().phi().value();
    uint32_t reg = RegionConnections->region(SiStripRegionCabling::Position(eta, phi));

    //Find subdet from det-id
    uint32_t subdet = static_cast<uint32_t>(SiStripRegionCabling::subdetFromDetId(idet.first));

    //Find layer from det-id
    uint32_t layer = tTopo.layer(idet.first);

    //@@ BELOW IS TEMP FIX TO HANDLE BUG IN DET CABLING
    const std::vector<const FedChannelConnection*>& conns = idet.second;

    //Update region cabling map
    regioncabling[reg][subdet][layer].push_back(SiStripRegionCabling::Element());
    auto& elem = regioncabling[reg][subdet][layer].back();
    elem.first = idet.first;
    elem.second.resize(conns.size());
    for (const auto& iconn : conns) {
      if ((iconn != nullptr) && (iconn->apvPairNumber() < conns.size())) {
        elem.second[iconn->apvPairNumber()] = *iconn;
      }
    }
  }

  //Add map to region cabling object
  RegionConnections->setRegionCabling(regioncabling);

  return RegionConnections;
}

DEFINE_FWK_EVENTSETUP_MODULE(SiStripRegionConnectivity);
