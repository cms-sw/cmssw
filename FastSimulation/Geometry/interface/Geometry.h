#ifndef FASTSIM_GEOMETRY_H
#define FASTSIM_GEOMETRY_H

#include "DataFormats/Math/interface/LorentzVector.h"
#include "FastSimulation/Layer/interface/ForwardLayer.h"
#include "FastSimulation/Layer/interface/BarrelLayer.h"

class GeometricSearchTracker;
class MagneticField;

#include <vector>

namespace edm { 
    class ParameterSet;
    class EventSetup;
}

namespace fastsim{
    class InteractionModel;
    class Geometry
    {
    public:
	/// Constructor
	Geometry(const edm::ParameterSet& cfg);

	/// Destructor
	~Geometry();

	void update(const edm::EventSetup & iSetup,const std::map<std::string,InteractionModel*> & interactionModelMap);

	// Returns the magnetic field
	double getMagneticFieldZ(const math::XYZTLorentzVector & position) const;

	const std::vector<std::unique_ptr<BarrelLayer> >& barrelLayers() const { return barrelLayers_; }
	const std::vector<std::unique_ptr<ForwardLayer> >& forwardLayers() const { return forwardLayers_; }
	
	double getMaxRadius() { return maxRadius_;}
	double getMaxZ() { return maxZ_;}
	

	friend std::ostream& operator << (std::ostream& o , const fastsim::Geometry & geometry); 
	
	// help to nagigate through layers
	const BarrelLayer * nextLayer(const BarrelLayer * layer) const
	{
	    if(layer == 0)
	    {
		return 0;
	    }
	    unsigned nextLayerIndex = layer->index() + 1;
	    return nextLayerIndex < barrelLayers_.size() ? barrelLayers_[nextLayerIndex].get() : 0;
	}

	const ForwardLayer * nextLayer(const ForwardLayer * layer) const
	{
	    if(layer == 0)
	    {
		return 0;
	    }
	    unsigned nextLayerIndex = layer->index() + 1;
	    return nextLayerIndex < forwardLayers_.size() ? forwardLayers_[nextLayerIndex].get() : 0;
	}

	const BarrelLayer * previousLayer(const BarrelLayer * layer) const
	{
	    if(layer == 0)
	    {
		return barrelLayers_.back().get();
	    }
	    return layer->index() > 0 ? barrelLayers_[layer->index() -1].get() : 0;
	}

	const ForwardLayer * previousLayer(const ForwardLayer * layer) const
	{
	    if(layer == 0)
	    {
		return forwardLayers_.back().get();
	    }
	    return layer->index() > 0 ? forwardLayers_[layer->index() -1].get() : 0;
	}

    private:

	std::vector<std::unique_ptr<BarrelLayer> >barrelLayers_;
	std::vector<std::unique_ptr<ForwardLayer> > forwardLayers_;
	std::unique_ptr<MagneticField> ownedMagneticField_;

	const MagneticField * magneticField_;
	const bool useFixedMagneticFieldZ_;
	const double fixedMagneticFieldZ_;
	const bool useTrackerRecoGeometryRecord_;
	const std::string trackerAlignmentLabel_;
	const std::vector<edm::ParameterSet> barrelLayerCfg_;
	const std::vector<edm::ParameterSet> forwardLayerCfg_;
	const double maxRadius_;
	const double maxZ_;
    };
    std::ostream& operator << (std::ostream& os , const fastsim::Geometry & geometry);
}


#endif
