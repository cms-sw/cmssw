#ifndef FASTSIM_BARRELLAYER_H
#define FASTSIM_BARRELLAYER_H

#include "FastSimulation/Layer/interface/Layer.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "TH1F.h"

namespace fastsim{

    class BarrelLayer : public Layer
    {
    public:
	~BarrelLayer(){};
	
	BarrelLayer(double radius) :
	    Layer(radius) {}
	
	BarrelLayer(BarrelLayer &&) = default;
	
	const double getRadius() const { return position_; }
	
	const double getThickness(const math::XYZTLorentzVector & position, const math::XYZTLorentzVector & momentum) const override
	{
	    if(!this->isOnSurface(position))
	    {
		throw cms::Exception("fastsim::BarrelLayer::getThickness") << "position is not on layer's surface";
	    }
	    double fabsCosTheta = fabs(momentum.Vect().Dot(position.Vect())) / momentum.Rho() / position.Rho();
	    return thicknessHist_->GetBinContent(thicknessHist_->GetXaxis()->FindBin(fabs(position.Z()))) / fabsCosTheta;
	}
	
	const double getMagneticFieldZ(const math::XYZTLorentzVector & position) const override
	{
	    if(!this->isOnSurface(position))
	    {
		throw cms::Exception("fastsim::BarrelLayer::getMagneticFieldZ") << "position is not on layer's surface";
	    }
	    return magneticFieldHist_->GetBinContent(magneticFieldHist_->GetXaxis()->FindBin(fabs(position.z())));
	}

	bool isForward() const override 
	{ 
	    return false;
	}

	bool isOnSurface(const math::XYZTLorentzVector & position) const override
	{
	    return fabs(position_ - sqrt(position.Perp2())) < epsilonDistanceR_;
	}
    };

}

#endif
