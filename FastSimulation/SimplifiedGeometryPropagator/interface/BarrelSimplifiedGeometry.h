#ifndef FASTSIM_BARRELSIMPLIFIEDGEOMETRY_H
#define FASTSIM_BARRELSIMPLIFIEDGEOMETRY_H

#include "FastSimulation/SimplifiedGeometryPropagator/interface/SimplifiedGeometry.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <TH1F.h>

///////////////////////////////////////////////
// Author: L. Vanelderen, S. Kurz
// Date: 29 May 2017
//////////////////////////////////////////////////////////

namespace fastsim {

  //! Implementation of a barrel detector layer (cylindrical).
  /*!
        A cylindrical layer with a given radius and a thickness (in radiation length).
        The layer is regarded infinitely long in Z-direction, however the thickness can vary (as a function of Z) and also be 0.
    */
  class BarrelSimplifiedGeometry : public SimplifiedGeometry {
  public:
    //! Constructor.
    /*!
            Create a barrel layer with a given radius.
            \param radius The radius of the layer (in cm).
        */
    BarrelSimplifiedGeometry(double radius) : SimplifiedGeometry(radius) {}

    //! Move constructor.
    BarrelSimplifiedGeometry(BarrelSimplifiedGeometry &&) = default;

    //! Default destructor.
    ~BarrelSimplifiedGeometry() override{};

    //! Return radius of the barrel layer.
    /*!
            \return The radius of the layer (in cm).
        */
    const double getRadius() const { return geomProperty_; }

    //! Return thickness of the barrel layer at a given position.
    /*!
            Returns the thickness of the barrel layer (in radiation length) at a specified position since the thickness can vary as a function of Z.
            \param position A position which has to be on the barrel layer.
            \return The thickness of the layer (in radiation length).
            \sa getThickness(const math::XYZTLorentzVector & position, const math::XYZTLorentzVector & momentum)
        */
    const double getThickness(const math::XYZTLorentzVector &position) const override {
      return thicknessHist_->GetBinContent(thicknessHist_->GetXaxis()->FindBin(fabs(position.Z())));
    }

    //! Return thickness of the barrel layer at a given position, also considering the incident angle.
    /*!
            Returns the thickness of the barrel layer (in radiation length) at a specified position and a given incident angle since the thickness can vary as a function of Z.
            \param position A position which has to be on the barrel layer.
            \param momentum The momentum of the incident particle.
            \return The thickness of the layer (in radiation length).
            \sa getThickness(const math::XYZTLorentzVector & position)
        */
    const double getThickness(const math::XYZTLorentzVector &position,
                              const math::XYZTLorentzVector &momentum) const override {
      // Do projection of norm(layer) on momentum vector
      // CosTheta = (momentum dot norm) / (length(momentum) / length(norm))
      return getThickness(position) /
             (fabs(momentum.X() * position.X() + momentum.Y() * position.Y()) /
              (momentum.P() * std::sqrt(position.X() * position.X() + position.Y() * position.Y())));
    }

    //! Return magnetic field (field only has Z component!) on the barrel layer.
    /*!
            Returns the magnetic field along the barrel layer at a specified position (radial symmetric).
            \param position A position which has to be on the barrel layer.
            \return The magnetic field on the layer.
        */
    const double getMagneticFieldZ(const math::XYZTLorentzVector &position) const override {
      return magneticFieldHist_->GetBinContent(magneticFieldHist_->GetXaxis()->FindBin(fabs(position.z())));
    }

    //! Returns false since class for barrel layer.
    /*!
            Function to easily destinguish barrel from forward layers (which both inherit from SimplifiedGeometry).
            \return false
        */
    bool isForward() const override { return false; }
  };

}  // namespace fastsim

#endif
