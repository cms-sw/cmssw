#ifndef FASTSIM_SIMPLIFIEDGEOMETRY_H
#define FASTSIM_SIMPLIFIEDGEOMETRY_H

#include "DataFormats/Math/interface/LorentzVector.h"

#include <memory>
#include <vector>

class DetLayer;
class MagneticField;
class GeometricSearchTracker;
class TH1F;

namespace edm
{
    class ParameterSet;
}

namespace fastsim
{
    class InteractionModel;
    class SimplifiedGeometryFactory;
    class SimplifiedGeometry
    {
    public:
	~SimplifiedGeometry();

	SimplifiedGeometry(double position);
	
	// Setters
	void setIndex(int index)
	{
	    index_ = index;
	}

	// Getters
	int index() const {return index_;}
	virtual const double getThickness(const math::XYZTLorentzVector & position, const math::XYZTLorentzVector & momentum) const = 0;
	const double getNuclearInteractionThicknessFactor() const {return nuclearInteractionThicknessFactor_; }
	const DetLayer* getDetLayer(double z = 0) const { return detLayer_; }
	virtual const double getMagneticFieldZ(const math::XYZTLorentzVector & position) const = 0;
	virtual bool isForward() const = 0;

	virtual bool isOnSurface(const math::XYZTLorentzVector & position) const = 0;

	const std::vector<InteractionModel *> & getInteractionModels() const
	{
	    return interactionModels_;
	}

	// friends
	friend std::ostream& operator << (std::ostream& os , const SimplifiedGeometry & layer);
	friend class fastsim::SimplifiedGeometryFactory;

    protected:
	
	// The position depends on which kind of layer is actually created
   	// BarrelSimplifiedGeometry: radius
    // ForwardSimplifiedGeometry: z
	double position_;
	double position2_;
	int index_;
	const DetLayer * detLayer_;
	std::unique_ptr<TH1F> magneticFieldHist_;
	std::unique_ptr<TH1F> thicknessHist_;
	double nuclearInteractionThicknessFactor_;
	std::vector<InteractionModel *> interactionModels_;
	
	static constexpr double epsilonDistanceZ_ = 1.0e-5;
	static constexpr double epsilonDistanceR_ = 1.0e-3;
    };

    std::ostream& operator << (std::ostream& os , const SimplifiedGeometry & layer);

}

#endif
