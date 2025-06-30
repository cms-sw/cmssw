//Framework Headers
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FastSimulation/SimplifiedGeometryPropagator/interface/SimplifiedGeometryFactory.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/ForwardSimplifiedGeometry.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/BarrelSimplifiedGeometry.h"

#include "MagneticField/UniformEngine/interface/UniformMagneticField.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/Geometry.h"

#include <iostream>
#include <map>
#include <memory>

using namespace fastsim;

GeometryConsumer::GeometryConsumer(const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC) :
  useFixedMagneticFieldZ(cfg.exists("magneticFieldZ")),
  useTrackerRecoGeometryRecord(cfg.getUntrackedParameter<bool>("useTrackerRecoGeometryRecord", true))
{
  if (useTrackerRecoGeometryRecord) {
    geometricSearchTrackerESToken = iC.esConsumes(edm::ESInputTag("", cfg.getUntrackedParameter<std::string>("trackerAlignmentLabel", "")));
  }
  if (!useFixedMagneticFieldZ) {
    magneticFieldESToken = iC.esConsumes();
  }
}

Geometry::~Geometry() { ; }

Geometry::Geometry(const edm::ParameterSet& cfg,
                   const std::vector<std::string>& interactionModelNames,
                   const edm::EventSetup& iSetup,
                   const GeometryConsumer& consumer)
    : cacheIdentifierTrackerRecoGeometry_(0),
      cacheIdentifierIdealMagneticField_(0),
      geometricSearchTracker_(nullptr),
      magneticField_(nullptr),
      fixedMagneticFieldZ_(cfg.getUntrackedParameter<double>("magneticFieldZ", 0.)),
      barrelLayerCfg_(cfg.getParameter<std::vector<edm::ParameterSet>>("BarrelLayers")),
      forwardLayerCfg_(cfg.getParameter<std::vector<edm::ParameterSet>>("EndcapLayers")),
      maxRadius_(cfg.getUntrackedParameter<double>("maxRadius", 500.)),
      maxZ_(cfg.getUntrackedParameter<double>("maxZ", 1200.)),
      barrelBoundary_(cfg.exists("trackerBarrelBoundary"))  // Hack to interface "old" calo to "new" tracking
      ,
      forwardBoundary_(cfg.exists("trackerForwardBoundary"))  // Hack to interface "old" calo to "new" tracking
      ,
      trackerBarrelBoundaryCfg_(barrelBoundary_
                                    ? cfg.getParameter<edm::ParameterSet>("trackerBarrelBoundary")
                                    : edm::ParameterSet())  // Hack to interface "old" calo to "new" tracking
      ,
      trackerForwardBoundaryCfg_(forwardBoundary_
                                     ? cfg.getParameter<edm::ParameterSet>("trackerForwardBoundary")
                                     : edm::ParameterSet())  // Hack to interface "old" calo to "new" tracking
{
  //----------------
  // find tracker reconstruction geometry
  //----------------
  if (iSetup.get<TrackerRecoGeometryRecord>().cacheIdentifier() != cacheIdentifierTrackerRecoGeometry_) {
    if (consumer.useTrackerRecoGeometryRecord) {
      geometricSearchTracker_ = &iSetup.getData(consumer.geometricSearchTrackerESToken);
    }
  }

  //----------------
  // update magnetic field
  //----------------
  if (iSetup.get<IdealMagneticFieldRecord>().cacheIdentifier() != cacheIdentifierIdealMagneticField_) {
    if (consumer.useFixedMagneticFieldZ)  // use constant magnetic field
    {
      ownedMagneticField_ = std::make_unique<UniformMagneticField>(fixedMagneticFieldZ_);
      magneticField_ = ownedMagneticField_.get();
    } else  // get magnetic field from EventSetup
    {
      magneticField_ = &iSetup.getData(consumer.magneticFieldESToken);
    }
  }

  //---------------
  // layer factory
  //---------------
  SimplifiedGeometryFactory simplifiedGeometryFactory(
      geometricSearchTracker_, *magneticField_, interactionModelNames, maxRadius_, maxZ_);

  //---------------
  // update barrel layers
  //---------------
  barrelLayers_.clear();
  for (const edm::ParameterSet& layerCfg : barrelLayerCfg_) {
    barrelLayers_.push_back(simplifiedGeometryFactory.createBarrelSimplifiedGeometry(layerCfg));
  }

  // Hack to interface "old" calo to "new" tracking
  if (barrelBoundary_) {
    barrelLayers_.push_back(simplifiedGeometryFactory.createBarrelSimplifiedGeometry(trackerBarrelBoundaryCfg_));
    barrelLayers_.back()->setCaloType(SimplifiedGeometry::TRACKERBOUNDARY);
  }

  for (unsigned index = 0; index < barrelLayers_.size(); index++) {
    // set index
    barrelLayers_[index]->setIndex(index);
    // check order
    if (index > 0) {
      if (barrelLayers_[index]->getRadius() <= barrelLayers_[index - 1]->getRadius()) {
        throw cms::Exception("fastsim::Geometry")
            << "barrel layers must be ordered according to increading radius"
            << "\nbarrel layer " << index << " has radius smaller than or equal to radius of barrel layer " << index - 1
            << " (" << barrelLayers_[index]->getRadius() << "/" << barrelLayers_[index - 1]->getRadius() << ")";
      }
    }
  }

  //--------------
  // update forward layers
  //--------------
  forwardLayers_.clear();
  for (const edm::ParameterSet& layerCfg : forwardLayerCfg_) {
    forwardLayers_.push_back(simplifiedGeometryFactory.createForwardSimplifiedGeometry(
        fastsim::SimplifiedGeometryFactory::POSFWD, layerCfg));
    forwardLayers_.insert(forwardLayers_.begin(),
                          simplifiedGeometryFactory.createForwardSimplifiedGeometry(
                              fastsim::SimplifiedGeometryFactory::NEGFWD, layerCfg));
  }

  // Hack to interface "old" calo to "new" tracking
  if (forwardBoundary_) {
    forwardLayers_.push_back(simplifiedGeometryFactory.createForwardSimplifiedGeometry(
        fastsim::SimplifiedGeometryFactory::POSFWD, trackerForwardBoundaryCfg_));
    forwardLayers_.back()->setCaloType(SimplifiedGeometry::TRACKERBOUNDARY);
    forwardLayers_.insert(forwardLayers_.begin(),
                          simplifiedGeometryFactory.createForwardSimplifiedGeometry(
                              fastsim::SimplifiedGeometryFactory::NEGFWD, trackerForwardBoundaryCfg_));
    forwardLayers_.front()->setCaloType(SimplifiedGeometry::TRACKERBOUNDARY);
  }

  for (unsigned index = 0; index < forwardLayers_.size(); index++) {
    // set index
    forwardLayers_[index]->setIndex(index);
    // check order
    if (index > 0) {
      if (forwardLayers_[index]->getZ() <= forwardLayers_[index - 1]->getZ()) {
        throw cms::Exception("fastsim::Geometry")
            << "forward layers must be ordered according to increasing z"
            << "forward layer " << index << " has z smaller than or equal to z of forward layer " << index - 1;
      }
    }
  }
}

bool Geometry::checkCache(const edm::EventSetup& iSetup) const {
  return iSetup.get<TrackerRecoGeometryRecord>().cacheIdentifier() == cacheIdentifierTrackerRecoGeometry_ &&
         iSetup.get<IdealMagneticFieldRecord>().cacheIdentifier() == cacheIdentifierIdealMagneticField_;
}

double fastsim::Geometry::getMagneticFieldZ(const math::XYZTLorentzVector& position) const {
  return magneticField_->inTesla(GlobalPoint(position.X(), position.Y(), position.Z())).z();
}

std::ostream& fastsim::operator<<(std::ostream& os, const fastsim::Geometry& geometry) {
  os << "-----------"
     << "\n# fastsim::Geometry"
     << "\n## BarrelLayers:";
  for (const auto& layer : geometry.barrelLayers_) {
    os << "\n   " << *layer << layer->getInteractionModelIndices().size() << " interaction models";
  }
  os << "\n## ForwardLayers:";
  for (const auto& layer : geometry.forwardLayers_) {
    os << "\n   " << *layer << layer->getInteractionModelIndices().size() << " interaction models";
  }
  os << "\n-----------";
  return os;
}
