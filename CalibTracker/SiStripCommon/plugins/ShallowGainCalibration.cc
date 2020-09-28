#include <memory>
#include "ShallowGainCalibration.h"

using namespace edm;
using namespace reco;
using namespace std;

ShallowGainCalibration::ShallowGainCalibration(const edm::ParameterSet& iConfig)
    : tracks_token_(consumes<edm::View<reco::Track>>(iConfig.getParameter<edm::InputTag>("Tracks"))),
      association_token_(consumes<TrajTrackAssociationCollection>(iConfig.getParameter<edm::InputTag>("Tracks"))),
      geomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
      gainToken_(esConsumes<SiStripGain, SiStripGainRcd>()),
      Suffix(iConfig.getParameter<std::string>("Suffix")),
      Prefix(iConfig.getParameter<std::string>("Prefix")) {
  produces<std::vector<int>>(Prefix + "trackindex" + Suffix);
  produces<std::vector<unsigned int>>(Prefix + "rawid" + Suffix);
  produces<std::vector<double>>(Prefix + "localdirx" + Suffix);
  produces<std::vector<double>>(Prefix + "localdiry" + Suffix);
  produces<std::vector<double>>(Prefix + "localdirz" + Suffix);
  produces<std::vector<unsigned short>>(Prefix + "firststrip" + Suffix);
  produces<std::vector<unsigned short>>(Prefix + "nstrips" + Suffix);
  produces<std::vector<bool>>(Prefix + "saturation" + Suffix);
  produces<std::vector<bool>>(Prefix + "overlapping" + Suffix);
  produces<std::vector<bool>>(Prefix + "farfromedge" + Suffix);
  produces<std::vector<unsigned int>>(Prefix + "charge" + Suffix);
  produces<std::vector<double>>(Prefix + "path" + Suffix);
#ifdef ExtendedCALIBTree
  produces<std::vector<double>>(Prefix + "chargeoverpath" + Suffix);
#endif
  produces<std::vector<unsigned char>>(Prefix + "amplitude" + Suffix);
  produces<std::vector<double>>(Prefix + "gainused" + Suffix);
  produces<std::vector<double>>(Prefix + "gainusedTick" + Suffix);
}

void ShallowGainCalibration::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  auto trackindex = std::make_unique<std::vector<int>>();
  auto rawid = std::make_unique<std::vector<unsigned int>>();
  auto localdirx = std::make_unique<std::vector<double>>();
  auto localdiry = std::make_unique<std::vector<double>>();
  auto localdirz = std::make_unique<std::vector<double>>();
  auto firststrip = std::make_unique<std::vector<unsigned short>>();
  auto nstrips = std::make_unique<std::vector<unsigned short>>();
  auto saturation = std::make_unique<std::vector<bool>>();
  auto overlapping = std::make_unique<std::vector<bool>>();
  auto farfromedge = std::make_unique<std::vector<bool>>();
  auto charge = std::make_unique<std::vector<unsigned int>>();
  auto path = std::make_unique<std::vector<double>>();
#ifdef ExtendedCALIBTree
  auto chargeoverpath = std::make_unique<std::vector<double>>();
#endif
  auto amplitude = std::make_unique<std::vector<unsigned char>>();
  auto gainused = std::make_unique<std::vector<double>>();
  auto gainusedTick = std::make_unique<std::vector<double>>();

  edm::ESHandle<SiStripGain> gainHandle = iSetup.getHandle(gainToken_);
  edm::Handle<edm::View<reco::Track>> tracks;
  iEvent.getByToken(tracks_token_, tracks);
  edm::Handle<TrajTrackAssociationCollection> associations;
  iEvent.getByToken(association_token_, associations);

  auto bundle = *runCache(iEvent.getRun().index());

  for (TrajTrackAssociationCollection::const_iterator association = associations->begin();
       association != associations->end();
       association++) {
    const Trajectory* traj = association->key.get();
    const reco::Track* track = association->val.get();

    vector<TrajectoryMeasurement> measurements = traj->measurements();
    for (vector<TrajectoryMeasurement>::const_iterator measurement_it = measurements.begin();
         measurement_it != measurements.end();
         measurement_it++) {
      TrajectoryStateOnSurface trajState = measurement_it->updatedState();
      if (!trajState.isValid())
        continue;

      const TrackingRecHit* hit = (*measurement_it->recHit()).hit();
      const SiStripRecHit1D* sistripsimple1dhit = dynamic_cast<const SiStripRecHit1D*>(hit);
      const SiStripRecHit2D* sistripsimplehit = dynamic_cast<const SiStripRecHit2D*>(hit);
      const SiStripMatchedRecHit2D* sistripmatchedhit = dynamic_cast<const SiStripMatchedRecHit2D*>(hit);
      const SiPixelRecHit* sipixelhit = dynamic_cast<const SiPixelRecHit*>(hit);

      const SiPixelCluster* PixelCluster = nullptr;
      const SiStripCluster* StripCluster = nullptr;
      uint32_t DetId = 0;

      for (unsigned int h = 0; h < 2; h++) {
        if (!sistripmatchedhit && h == 1) {
          continue;
        } else if (sistripmatchedhit && h == 0) {
          StripCluster = &sistripmatchedhit->monoCluster();
          DetId = sistripmatchedhit->monoId();
        } else if (sistripmatchedhit && h == 1) {
          StripCluster = &sistripmatchedhit->stereoCluster();
          ;
          DetId = sistripmatchedhit->stereoId();
        } else if (sistripsimplehit) {
          StripCluster = (sistripsimplehit->cluster()).get();
          DetId = sistripsimplehit->geographicalId().rawId();
        } else if (sistripsimple1dhit) {
          StripCluster = (sistripsimple1dhit->cluster()).get();
          DetId = sistripsimple1dhit->geographicalId().rawId();
        } else if (sipixelhit) {
          PixelCluster = (sipixelhit->cluster()).get();
          DetId = sipixelhit->geographicalId().rawId();
        } else {
          continue;
        }

        LocalVector trackDirection = trajState.localDirection();
        double cosine = trackDirection.z() / trackDirection.mag();
        bool Saturation = false;
        bool Overlapping = false;
        unsigned int Charge = 0;
        auto thicknessMap = bundle.getThicknessMap();
        double Path = (10.0 * thicknessMap[DetId]) / fabs(cosine);
        double PrevGain = -1;
        double PrevGainTick = -1;
        int FirstStrip = 0;
        int NStrips = 0;

        if (StripCluster) {
          const auto& Ampls = StripCluster->amplitudes();
          FirstStrip = StripCluster->firstStrip();
          NStrips = Ampls.size();
          int APVId = FirstStrip / 128;

          if (gainHandle.isValid()) {
            PrevGain = gainHandle->getApvGain(APVId, gainHandle->getRange(DetId, 1), 1);
            PrevGainTick = gainHandle->getApvGain(APVId, gainHandle->getRange(DetId, 0), 1);
          }

          for (unsigned int a = 0; a < Ampls.size(); a++) {
            Charge += Ampls[a];
            if (Ampls[a] >= 254)
              Saturation = true;
            amplitude->push_back(Ampls[a]);
          }

          if (FirstStrip == 0)
            Overlapping = true;
          if (FirstStrip == 128)
            Overlapping = true;
          if (FirstStrip == 256)
            Overlapping = true;
          if (FirstStrip == 384)
            Overlapping = true;
          if (FirstStrip == 512)
            Overlapping = true;
          if (FirstStrip == 640)
            Overlapping = true;

          if (FirstStrip <= 127 && FirstStrip + Ampls.size() > 127)
            Overlapping = true;
          if (FirstStrip <= 255 && FirstStrip + Ampls.size() > 255)
            Overlapping = true;
          if (FirstStrip <= 383 && FirstStrip + Ampls.size() > 383)
            Overlapping = true;
          if (FirstStrip <= 511 && FirstStrip + Ampls.size() > 511)
            Overlapping = true;
          if (FirstStrip <= 639 && FirstStrip + Ampls.size() > 639)
            Overlapping = true;

          if (FirstStrip + Ampls.size() == 127)
            Overlapping = true;
          if (FirstStrip + Ampls.size() == 255)
            Overlapping = true;
          if (FirstStrip + Ampls.size() == 383)
            Overlapping = true;
          if (FirstStrip + Ampls.size() == 511)
            Overlapping = true;
          if (FirstStrip + Ampls.size() == 639)
            Overlapping = true;
          if (FirstStrip + Ampls.size() == 767)
            Overlapping = true;
        } else if (PixelCluster) {
          const auto& Ampls = PixelCluster->pixelADC();
          int FirstRow = PixelCluster->minPixelRow();
          int FirstCol = PixelCluster->minPixelCol();
          FirstStrip = ((FirstRow / 80) << 3 | (FirstCol / 52)) * 128;  //Hack to save the APVId
          NStrips = 0;
          Saturation = false;
          Overlapping = false;

          for (unsigned int a = 0; a < Ampls.size(); a++) {
            Charge += Ampls[a];
            if (Ampls[a] >= 254)
              Saturation = true;
          }
        }
#ifdef ExtendedCALIBTree
        double ChargeOverPath = (double)Charge / Path;
#endif

        trackindex->push_back(shallow::findTrackIndex(tracks, track));
        rawid->push_back(DetId);
        localdirx->push_back(trackDirection.x());
        localdiry->push_back(trackDirection.y());
        localdirz->push_back(trackDirection.z());
        firststrip->push_back(FirstStrip);
        nstrips->push_back(NStrips);
        saturation->push_back(Saturation);
        overlapping->push_back(Overlapping);
        farfromedge->push_back(StripCluster ? isFarFromBorder(&trajState, DetId, bundle.getTrackerGeometry()) : true);
        charge->push_back(Charge);
        path->push_back(Path);
#ifdef ExtendedCALIBTree
        chargeoverpath->push_back(ChargeOverPath);
#endif
        gainused->push_back(PrevGain);
        gainusedTick->push_back(PrevGainTick);
      }
    }
  }

  iEvent.put(std::move(trackindex), Prefix + "trackindex" + Suffix);
  iEvent.put(std::move(rawid), Prefix + "rawid" + Suffix);
  iEvent.put(std::move(localdirx), Prefix + "localdirx" + Suffix);
  iEvent.put(std::move(localdiry), Prefix + "localdiry" + Suffix);
  iEvent.put(std::move(localdirz), Prefix + "localdirz" + Suffix);
  iEvent.put(std::move(firststrip), Prefix + "firststrip" + Suffix);
  iEvent.put(std::move(nstrips), Prefix + "nstrips" + Suffix);
  iEvent.put(std::move(saturation), Prefix + "saturation" + Suffix);
  iEvent.put(std::move(overlapping), Prefix + "overlapping" + Suffix);
  iEvent.put(std::move(farfromedge), Prefix + "farfromedge" + Suffix);
  iEvent.put(std::move(charge), Prefix + "charge" + Suffix);
  iEvent.put(std::move(path), Prefix + "path" + Suffix);
#ifdef ExtendedCALIBTree
  iEvent.put(std::move(chargeoverpath), Prefix + "chargeoverpath" + Suffix);
#endif
  iEvent.put(std::move(amplitude), Prefix + "amplitude" + Suffix);
  iEvent.put(std::move(gainused), Prefix + "gainused" + Suffix);
  iEvent.put(std::move(gainusedTick), Prefix + "gainusedTick" + Suffix);
}

bool ShallowGainCalibration::isFarFromBorder(TrajectoryStateOnSurface* trajState,
                                             const uint32_t detid,
                                             const TrackerGeometry* tkGeom) const {
  LocalPoint HitLocalPos = trajState->localPosition();
  LocalError HitLocalError = trajState->localError().positionError();

  const GeomDetUnit* it = tkGeom->idToDetUnit(DetId(detid));
  if (dynamic_cast<const StripGeomDetUnit*>(it) == nullptr && dynamic_cast<const PixelGeomDetUnit*>(it) == nullptr) {
    throw cms::Exception("Logic Error") << "\t\t this detID doesn't seem to belong to the Tracker";
  }

  const BoundPlane plane = it->surface();
  const TrapezoidalPlaneBounds* trapezoidalBounds(dynamic_cast<const TrapezoidalPlaneBounds*>(&(plane.bounds())));
  const RectangularPlaneBounds* rectangularBounds(dynamic_cast<const RectangularPlaneBounds*>(&(plane.bounds())));

  double DistFromBorder = 1.0;
  double HalfLength = it->surface().bounds().length() / 2.0;

  if (trapezoidalBounds) {
    std::array<const float, 4> const& parameters = (*trapezoidalBounds).parameters();
    HalfLength = parameters[3];
  } else if (rectangularBounds) {
    HalfLength = it->surface().bounds().length() / 2.0;
  } else {
    return false;
  }

  if (fabs(HitLocalPos.y()) + HitLocalError.yy() >= (HalfLength - DistFromBorder))
    return false;

  return true;
}

std::shared_ptr<shallowGainCalibration::bundle> ShallowGainCalibration::globalBeginRun(
    edm::Run const& iRun, edm::EventSetup const& iSetup) const {
  edm::ESHandle<TrackerGeometry> theTrackerGeometry = iSetup.getHandle(geomToken_);
  auto bundle = std::make_shared<shallowGainCalibration::bundle>(theTrackerGeometry.product());

  for (const auto& it : theTrackerGeometry->detUnits()) {
    double detThickness = 1.;
    auto id = (it->geographicalId()).rawId();

    bool isPixel = dynamic_cast<const PixelGeomDetUnit*>(it) != nullptr;
    bool isStrip = dynamic_cast<const StripGeomDetUnit*>(it) != nullptr;

    if (!isPixel && !isStrip) {
      throw cms::Exception("Logic Error") << "\t\t this detID doesn't seem to belong to the Tracker";
    } else {
      detThickness = it->surface().bounds().thickness();
    }
    bundle.get()->updateMap(id, detThickness);
  }
  return bundle;
}

void ShallowGainCalibration::globalEndRun(edm::Run const&, edm::EventSetup const&) const {}
