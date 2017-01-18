//Framework Headers
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"


#include "FastSimulation/Layer/interface/LayerFactory.h"
#include "FastSimulation/Layer/interface/ForwardLayer.h"
#include "FastSimulation/Layer/interface/BarrelLayer.h"


#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/UniformEngine/src/UniformMagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FastSimulation/Geometry/interface/Geometry.h"

#include <iostream>
#include <map>

using namespace fastsim;

Geometry::~Geometry(){;}

Geometry::Geometry(const edm::ParameterSet& cfg)
    : magneticField_(0)
    , useFixedMagneticFieldZ_(cfg.exists("magneticFieldZ"))
    , fixedMagneticFieldZ_(cfg.getUntrackedParameter<double>("magneticFieldZ",0.))
    , useTrackerRecoGeometryRecord_(cfg.getUntrackedParameter<bool>("useTrackerRecoGeometryRecord",true))
    , trackerAlignmentLabel_(cfg.getUntrackedParameter<std::string>("trackerAlignmentLabel",""))
    , barrelLayerCfg_(cfg.getParameter<std::vector<edm::ParameterSet>>("BarrelLayers"))
    , forwardLayerCfg_(cfg.getParameter<std::vector<edm::ParameterSet>>("ForwardLayers"))
    , maxRadius_(cfg.getUntrackedParameter<double>("maxRadius",240.))
    , maxZ_(cfg.getUntrackedParameter<double>("maxZ",600.))
{};

void Geometry::update(const edm::EventSetup & iSetup,const std::map<std::string,fastsim::InteractionModel*> & interactionModelMap)
{

    //----------------
    // find tracker reconstruction geometry
    //----------------
    const GeometricSearchTracker * geometricSearchTracker = 0;
    if(useTrackerRecoGeometryRecord_)
    {
	edm::ESHandle<GeometricSearchTracker> geometricSearchTrackerHandle;
	iSetup.get<TrackerRecoGeometryRecord>().get(trackerAlignmentLabel_,geometricSearchTrackerHandle);
	geometricSearchTracker = &(*geometricSearchTrackerHandle);
    }

    //----------------
    // update magnetic field
    //----------------
    if(useFixedMagneticFieldZ_)
    {
	ownedMagneticField_.reset(new UniformMagneticField(fixedMagneticFieldZ_));
	magneticField_ = ownedMagneticField_.get();
    }
    else
    {
	edm::ESHandle<MagneticField> magneticField;
	iSetup.get<IdealMagneticFieldRecord>().get(magneticField);
	magneticField_ = &(*magneticField);
    }

    //---------------
    // layer factory
    //---------------
    fastsim::LayerFactory layerFactory(geometricSearchTracker
				       ,*magneticField_
				       ,interactionModelMap
				       ,maxRadius_
				       ,maxZ_);
    //---------------
    // update barrel layers
    //---------------
    barrelLayers_.clear();
    for(const edm::ParameterSet & layerCfg : barrelLayerCfg_)
    {
	barrelLayers_.push_back(layerFactory.createBarrelLayer(layerCfg));
    }
    for(unsigned index = 0;index < barrelLayers_.size();index++)
    {
	// set index
	barrelLayers_[index]->setIndex(index);
	// check order
	if(index > 0)
	{
	    if(barrelLayers_[index]->getRadius() <= barrelLayers_[index-1]->getRadius())
	    {
		throw cms::Exception("fastsim::Geometry") 
		    << "barrel layers must be ordered according to increading radius"
		    << "\nbarrel layer " << index 
		    << " has radius smaller than or equal to radius of barrel layer " << index -1
            << " (" << barrelLayers_[index]->getRadius() << "/" << barrelLayers_[index-1]->getRadius() << ")";
	    }
	}
    }
    
    //--------------
    // update forward layers
    //--------------
    forwardLayers_.clear();
    for(const edm::ParameterSet & layerCfg : forwardLayerCfg_)
    {
	forwardLayers_.push_back(layerFactory.createForwardLayer(fastsim::LayerFactory::POSFWD,layerCfg));
	forwardLayers_.insert(forwardLayers_.begin(),layerFactory.createForwardLayer(fastsim::LayerFactory::NEGFWD,layerCfg));
    }
    for(unsigned index = 0;index < forwardLayers_.size();index++)
    {
	// set index
	forwardLayers_[index]->setIndex(index);
	// check order
	if(index > 0)
	{
	    if(forwardLayers_[index]->getZ() <= forwardLayers_[index-1]->getZ())
	    {
		throw cms::Exception("fastsim::Geometry") 
		    << "forward layers must be ordered according to increasing z"
		    << "forward layer " << index 
		    << " has z smaller than or equal to z of forward layer " << index -1;
	    }
	}
    }
}

double fastsim::Geometry::getMagneticFieldZ (const math::XYZTLorentzVector & position) const
{
    return magneticField_->inTesla(GlobalPoint(position.X(),position.Y(),position.Z())).z();
}

std::ostream& fastsim::operator << (std::ostream& os , const fastsim::Geometry & geometry)
{
    os << "-----------"
       << "\n# fastsim::Geometry"
       << "\n## BarrelLayers:";
    for(const auto & layer : geometry.barrelLayers_)
    {
	os << "\n   " << *layer
	   << layer->getInteractionModels().size() << " interaction models";
    }
    os << "\n## ForwardLayers:";
    for(const auto & layer : geometry.forwardLayers_)
    {
	os << "\n   " << *layer
	   << layer->getInteractionModels().size() << " interaction models";
    }
    os << "\n-----------";
    return os;
}

