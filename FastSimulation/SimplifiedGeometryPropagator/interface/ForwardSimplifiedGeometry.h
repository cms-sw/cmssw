#ifndef FASTSIM_FORWARDSIMPLIFIEDGEOMETRY_H
#define FASTSIM_FORWARDSIMPLIFIEDGEOMETRY_H

#include "FastSimulation/SimplifiedGeometryPropagator/interface/SimplifiedGeometry.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "TH1F.h"
#include "FWCore/Utilities/interface/Exception.h"


namespace fastsim{

    class ForwardSimplifiedGeometry : public SimplifiedGeometry
    {
    public:
	~ForwardSimplifiedGeometry(){};

	ForwardSimplifiedGeometry(double z) :
	    SimplifiedGeometry(z) {}

	const double getZ() const { return position_; }

	const double getThickness(const math::XYZTLorentzVector & position, const math::XYZTLorentzVector & momentum) const override
	{
	    if(!this->isOnSurface(position))
	    {
		return 0;
	    }
	    return thicknessHist_->GetBinContent(thicknessHist_->GetXaxis()->FindBin(fabs(position.Pt()))) / fabs(momentum.Pz()) * momentum.P();
	}

	const double getMagneticFieldZ(const math::XYZTLorentzVector & position) const override
	{
	    if(!this->isOnSurface(position))
	    {
		throw cms::Exception("fastsim::BarrelSimplifiedGeometry::getMagneticFieldZ") << "position is not on layer's surface";
	    }
	    return magneticFieldHist_->GetBinContent(magneticFieldHist_->GetXaxis()->FindBin(position.Pt()));
	}
	
	bool isForward() const override {return true;}
	
	bool isOnSurface(const math::XYZTLorentzVector & position) const override
	{
	    return fabs(position_ - position.Z()) < epsilonDistanceZ_;
	}

    };

}

#endif
