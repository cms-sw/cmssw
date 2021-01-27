#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/Common/interface/View.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"

#include <DataFormats/GeometryVector/interface/GlobalPoint.h>
#include <Geometry/CaloGeometry/interface/CaloGeometry.h>
#include <Geometry/HGCalGeometry/interface/HGCalGeometry.h>
#include <Geometry/Records/interface/CaloGeometryRecord.h>
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

#include <vector>
#include <iostream>

template <typename T>
class PositionFromDetIDTableProducer : public edm::stream::EDProducer<> {
public:
  PositionFromDetIDTableProducer(edm::ParameterSet const& params)
      : name_(params.getParameter<std::string>("name")),
        doc_(params.getParameter<std::string>("doc")),
        src_(consumes<T>(params.getParameter<edm::InputTag>("src"))),
        cut_(params.getParameter<std::string>("cut"), true) {
    produces<nanoaod::FlatTable>();
  }

  ~PositionFromDetIDTableProducer() override {}

  void beginRun(const edm::Run&, const edm::EventSetup& iSetup) {
      // TODO: check that the geometry exists
      iSetup.get<CaloGeometryRecord>().get(caloGeom_);
      iSetup.get<GlobalTrackingGeometryRecord>().get(trackGeom_);
  }

  GlobalPoint positionFromHit(const PCaloHit& hit) {
    DetId id = hit.id();
    return positionFromDetId(id);
  }

  GlobalPoint positionFromHit(const CaloRecHit& hit) {
    return positionFromDetId(hit.detid());
  }

  GlobalPoint positionFromDetId(DetId id) {
    DetId::Detector det = id.det();
    int subid = (det == DetId::HGCalEE || det == DetId::HGCalHSi || det == DetId::HGCalHSc)
                   ? ForwardSubdetector::ForwardEmpty
                   : id.subdetId();
    auto geom = caloGeom_->getSubdetectorGeometry(det, subid);

    GlobalPoint position;
    if (id.det() == DetId::Hcal) {
      position = geom->getGeometry(id)->getPosition();
    } else if (id.det() == DetId::HGCalEE || id.det() == DetId::HGCalHSi || id.det() == DetId::HGCalHSc) {
      auto hg = static_cast<const HGCalGeometry*>(geom);
      position = hg->getPosition(id);
    }
    else {
        throw cms::Exception("PositionFromDetIDTableProducer") << "Unsupported DetId type";
    }

    return position;
  }

  GlobalPoint positionFromHit(const PSimHit& hit) {
      auto surface = trackGeom_->idToDet(hit.detUnitId())->surface();
      //LocalPoint localPos = surface.position();
      GlobalPoint position = surface.toGlobal(hit.localPosition());
      return position;
  }

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override {
    edm::Handle<T> objs;
    iEvent.getByToken(src_, objs);

    std::vector<float> xvals;
    std::vector<float> yvals;
    std::vector<float> zvals;
    for (const auto& obj : *objs) {
      if (cut_(obj)) {
        auto position = positionFromHit(obj);
        xvals.emplace_back(position.x());
        yvals.emplace_back(position.y());
        zvals.emplace_back(position.z());
      }
    }

    auto tab = std::make_unique<nanoaod::FlatTable>(xvals.size(), name_, false, true);
    tab->addColumn<float>("x", xvals, "x position");
    tab->addColumn<float>("y", yvals, "y position");
    tab->addColumn<float>("z", zvals, "z position");

    iEvent.put(std::move(tab));
  }

protected:
  const std::string name_, doc_;
  const edm::EDGetTokenT<T> src_;
  const StringCutObjectSelector<typename T::value_type> cut_;
  edm::ESHandle<CaloGeometry> caloGeom_;
  edm::ESHandle<GlobalTrackingGeometry> trackGeom_;

};

#include "FWCore/Framework/interface/MakerMacros.h"
typedef PositionFromDetIDTableProducer<std::vector<PCaloHit>> PCaloHitPositionFromDetIDTableProducer;
typedef PositionFromDetIDTableProducer<std::vector<PSimHit>> PSimHitPositionFromDetIDTableProducer;
typedef PositionFromDetIDTableProducer<HGCRecHitCollection> HGCRecHitPositionFromDetIDTableProducer;
DEFINE_FWK_MODULE(HGCRecHitPositionFromDetIDTableProducer);
DEFINE_FWK_MODULE(PCaloHitPositionFromDetIDTableProducer);
DEFINE_FWK_MODULE(PSimHitPositionFromDetIDTableProducer);
