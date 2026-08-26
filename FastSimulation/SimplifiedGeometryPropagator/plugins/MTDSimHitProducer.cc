////////////////////////////////////////////////////////////////////////////////
///
/// MTD fastsim plugin
///
/// Author: H. Jeong at Aug 2026
///
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// Headers
////////////////////////////////////////////////////////////////////////////////
//------------------------------------------------------------------------------
// framework
//------------------------------------------------------------------------------
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//------------------------------------------------------------------------------
// fastsim
//------------------------------------------------------------------------------
#include "FastSimulation/SimplifiedGeometryPropagator/interface/InteractionModel.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/InteractionModelFactory.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/Particle.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/SimplifiedGeometry.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/Constants.h"
#include "FastSimulation/TrajectoryManager/interface/InsideBoundsMeasurementEstimator.h"

//------------------------------------------------------------------------------
// data formats
//------------------------------------------------------------------------------
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

//------------------------------------------------------------------------------
// MTD geometry
//------------------------------------------------------------------------------
#include "RecoMTD/DetLayers/interface/MTDDetLayerGeometry.h"
#include "RecoMTD/DetLayers/interface/MTDTrayBarrelLayer.h"
#include "RecoMTD/DetLayers/interface/MTDSectorForwardDoubleLayer.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/UniformEngine/interface/UniformMagneticField.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"

//------------------------------------------------------------------------------
// standard
//------------------------------------------------------------------------------
#include <cmath>
#include <string>
#include <vector>
#include <memory>

namespace fastsim {
  ////////////////////////////////////////////////////////////////////////////////
  /// Declare class
  ////////////////////////////////////////////////////////////////////////////////
  class MTDSimHitProducer : public InteractionModel {
  public:
    MTDSimHitProducer(const std::string& name, const edm::ParameterSet& cfg);
    ~MTDSimHitProducer() override = default;

    void interact(Particle& particle,
                  const SimplifiedGeometry& layer,
                  std::vector<std::unique_ptr<Particle>>& secondaries,
                  const RandomEngineAndDistribution& random) override;

    void registerProducts(edm::ProducesCollector producesCollector) const override;
    void storeProducts(edm::Event& iEvent) override;

  private:
    //------------------------------------------------------------------------------
    // geometry parameters (fallback when DetLayer not available)
    //------------------------------------------------------------------------------
    const double btlRadius_;
    const double btlHalfLength_;
    const double etlRMin_;
    const double etlRMax_;

    //------------------------------------------------------------------------------
    // hit containers
    //------------------------------------------------------------------------------
    std::unique_ptr<edm::PSimHitContainer> btlHits_;
    std::unique_ptr<edm::PSimHitContainer> etlHits_;
  };

  ////////////////////////////////////////////////////////////////////////////////
  /// Constructor
  ////////////////////////////////////////////////////////////////////////////////
  MTDSimHitProducer::MTDSimHitProducer(const std::string& name, const edm::ParameterSet& cfg)
    : InteractionModel(name),
      btlRadius_(cfg.getParameter<double>("btlRadius")),
      btlHalfLength_(cfg.getParameter<double>("btlHalfLength")),
      etlRMin_(cfg.getParameter<double>("etlRMin")),
      etlRMax_(cfg.getParameter<double>("etlRMax")),
      btlHits_(std::make_unique<edm::PSimHitContainer>()),
      etlHits_(std::make_unique<edm::PSimHitContainer>()) {}

  ////////////////////////////////////////////////////////////////////////////////
  /// Register the SimHit collection
  ////////////////////////////////////////////////////////////////////////////////
  void MTDSimHitProducer::registerProducts(edm::ProducesCollector producesCollector) const {
    producesCollector.produces<edm::PSimHitContainer>("FastTimerHitsBarrel");
    producesCollector.produces<edm::PSimHitContainer>("FastTimerHitsEndcap");
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Store the SimHit collection
  ////////////////////////////////////////////////////////////////////////////////
  void MTDSimHitProducer::storeProducts(edm::Event& iEvent) {
    iEvent.put(std::move(btlHits_), "FastTimerHitsBarrel");
    iEvent.put(std::move(etlHits_), "FastTimerHitsEndcap");
    btlHits_ = std::make_unique<edm::PSimHitContainer>();
    etlHits_ = std::make_unique<edm::PSimHitContainer>();
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Perform the interaction.
  ///
  /// \param particle The particle that interacts with the matter.
  /// \param layer The detector layer that interacts with the particle.
  /// \param secondaries Particles that are produced in the interaction (if any).
  /// \param random The Random Engine.
  ////////////////////////////////////////////////////////////////////////////////
  void MTDSimHitProducer::interact(Particle& particle,
                                   const SimplifiedGeometry& layer,
                                   std::vector<std::unique_ptr<Particle>>& secondaries,
                                   const RandomEngineAndDistribution& random) {
    //------------------------------------------------------------------------------
    // Not interested in the neutral
    //------------------------------------------------------------------------------
    if (particle.charge() == 0)
      return;

    //------------------------------------------------------------------------------
    // No material
    //------------------------------------------------------------------------------
    if (layer.getThickness(particle.position(), particle.momentum()) < 1E-10)
      return;

    //------------------------------------------------------------------------------
    // Read position and momentum
    //------------------------------------------------------------------------------
    const double x = particle.position().X();
    const double y = particle.position().Y();
    const double z = particle.position().Z();
    const double t = particle.position().T();
    const double px = particle.momentum().X();
    const double py = particle.momentum().Y();
    const double pz = particle.momentum().Z();
    const double p = particle.momentum().P();

    //------------------------------------------------------------------------------
    // Not interested in the rest
    //------------------------------------------------------------------------------
    if (p == 0.)
      return;

    //------------------------------------------------------------------------------
    // simTrackId: use mother if there is
    //------------------------------------------------------------------------------
    const int simTrackId =
      particle.getMotherSimTrackIndex() >= 0 ? particle.getMotherSimTrackIndex() : particle.simTrackIndex();

    //------------------------------------------------------------------------------
    // energy deposit from EnergyLoss interaction model
    //------------------------------------------------------------------------------
    const double energyDeposit = particle.getEnergyDeposit();
    particle.setEnergyDeposit(0);

    //------------------------------------------------------------------------------
    // Allocate DetId
    // Use layer . getDetLayer() if it is, or 0
    //------------------------------------------------------------------------------
    uint32_t detUnitId = 0;
    Local3DPoint entryPoint, exitPoint;

    if (!layer.isForward()) {
      //----------------------------------------------------------
      // BTL: barrel layer
      //----------------------------------------------------------
      if (std::abs(z) > btlHalfLength_)
        return;

      const DetLayer* detLayer = layer.getDetLayer();

      if (detLayer) {
        //--------------------------------------
        // Construct TrajectoryStateOnSurface
        //--------------------------------------
        UniformMagneticField magneticField(layer.getMagneticFieldZ(particle.position()));
        GlobalPoint position(x, y, z);
        GlobalVector momentum(px, py, pz);
        auto plane = detLayer->surface().tangentPlane(position);
        TrajectoryStateOnSurface tsos(GlobalTrajectoryParameters(position, momentum, TrackCharge(particle.charge()), &magneticField), *plane);

        //--------------------------------------
        // Fint the nearest DetUnit
        //--------------------------------------
        AnalyticalPropagator propagator(&magneticField, anyDirection);
        InsideBoundsMeasurementEstimator est;
        auto compatDets = detLayer->compatibleDets(tsos, propagator, est);
        if (!compatDets.empty()) {
          const GeomDet* det = compatDets.front().first;
          detUnitId = det->geographicalId().rawId();

          // transform to the local frame
          LocalPoint localPos = det->toLocal(position);
          entryPoint = localPos;
          exitPoint = localPos;
        }
      } else {
        //--------------------------------------
        // when failed to find DetLayer, use global frame
        //--------------------------------------
        entryPoint = Local3DPoint(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
        exitPoint = entryPoint;
      }

      btlHits_->emplace_back(entryPoint,
                             exitPoint,
                             static_cast<float>(p),
                             static_cast<float>(t),
                             static_cast<float>(energyDeposit),
                             particle.pdgId(),
                             detUnitId,
                             simTrackId,
                             0.f,
                             0.f);
    } else {
      //----------------------------------------------------------
      // ETL: forward layer
      //----------------------------------------------------------
      const double r = std::sqrt(x * x + y * y);
      if (r < etlRMin_ || r > etlRMax_)
        return;

      const DetLayer* detLayer = layer.getDetLayer();

      if (detLayer) {
        UniformMagneticField magneticField(layer.getMagneticFieldZ(particle.position()));
        GlobalPoint position(x, y, z);
        GlobalVector momentum(px, py, pz);
        auto plane = detLayer->surface().tangentPlane(position);
        TrajectoryStateOnSurface tsos(GlobalTrajectoryParameters(position, momentum, TrackCharge(particle.charge()), &magneticField), *plane);

        AnalyticalPropagator propagator(&magneticField, anyDirection);
        InsideBoundsMeasurementEstimator est;
        auto compatDets = detLayer->compatibleDets(tsos, propagator, est);
        if (!compatDets.empty()) {
          const GeomDet* det = compatDets.front().first;
          detUnitId = det->geographicalId().rawId();

          LocalPoint localPos = det->toLocal(position);
          entryPoint = localPos;
          exitPoint = localPos;
        }
      } else {
        entryPoint = Local3DPoint(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
        exitPoint = entryPoint;
      }

      etlHits_->emplace_back(entryPoint,
                             exitPoint,
                             static_cast<float>(p),
                             static_cast<float>(t),
                             static_cast<float>(energyDeposit),
                             particle.pdgId(),
                             detUnitId,
                             simTrackId,
                             0.f,
                             0.f);
    }
  }
}  // namespace fastsim

DEFINE_EDM_PLUGIN(fastsim::InteractionModelFactory, fastsim::MTDSimHitProducer, "fastsim::MTDSimHitProducer");
