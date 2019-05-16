#ifndef FASTSIM_FORWARDSIMPLIFIEDGEOMETRY_H
#define FASTSIM_FORWARDSIMPLIFIEDGEOMETRY_H

#include "FastSimulation/SimplifiedGeometryPropagator/interface/SimplifiedGeometry.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include <TH1F.h>
#include "FWCore/Utilities/interface/Exception.h"

///////////////////////////////////////////////
// Author: L. Vanelderen, S. Kurz
// Date: 29 May 2017
//////////////////////////////////////////////////////////

namespace fastsim {

  //! Implementation of a forward detector layer (disk).
  /*!
        A disk with a position in Z and a thickness (in radiation length).
        The layer is regarded to have an infinite radius, however the thickness can vary (as a function of the radius) and also be 0.
    */
  class ForwardSimplifiedGeometry : public SimplifiedGeometry {
  public:
    //! Constructor.
    /*!
            Create a forward layer with a given position (along z-axis).
            \param radius The z-position of the layer (in cm).
        */
    ForwardSimplifiedGeometry(double z) : SimplifiedGeometry(z) {}

    //! Move constructor.
    ForwardSimplifiedGeometry(ForwardSimplifiedGeometry &&) = default;

    //! Default destructor.
    ~ForwardSimplifiedGeometry() override{};

    //! Return z-position of the forward layer.
    /*!
            \return The z-position of the layer (in cm).
        */
    const double getZ() const { return geomProperty_; }

    //! Return thickness of the forward layer at a given position.
    /*!
            Returns the thickness of the forward layer (in radiation length) at a specified position since the thickness can vary as a function of the radius.
            \param position A position which has to be on the forward layer.
            \return The thickness of the layer (in radiation length).
            \sa getThickness(const math::XYZTLorentzVector & position, const math::XYZTLorentzVector & momentum)
        */
    const double getThickness(const math::XYZTLorentzVector &position) const override {
      return thicknessHist_->GetBinContent(thicknessHist_->GetXaxis()->FindBin(position.Pt()));
    }

    //! Return thickness of the forward layer at a given position, also considering the incident angle.
    /*!
            Returns the thickness of the forward layer (in radiation length) at a specified position and a given incident angle since the thickness can vary as a function of the radius.
            \param position A position which has to be on the forward layer.
            \param momentum The momentum of the incident particle.
            \return The thickness of the layer (in radiation length).
            \sa getThickness(const math::XYZTLorentzVector & position)
        */
    const double getThickness(const math::XYZTLorentzVector &position,
                              const math::XYZTLorentzVector &momentum) const override {
      return getThickness(position) / fabs(momentum.Pz()) * momentum.P();
    }

    //! Return magnetic field (field only has Z component!) on the forward layer.
    /*!
            Returns the magnetic field along the forward layer at a specified position (radial symmetric).
            \param position A position which has to be on the forward layer.
            \return The magnetic field on the layer.
        */
    const double getMagneticFieldZ(const math::XYZTLorentzVector &position) const override {
      return magneticFieldHist_->GetBinContent(magneticFieldHist_->GetXaxis()->FindBin(position.Pt()));
    }

    //! Returns true since class for forward layer.
    /*!
            Function to easily destinguish barrel from forward layers (which both inherit from ForwardSimplifiedGeometry).
            \return true
        */
    bool isForward() const override { return true; }
  };

}  // namespace fastsim

#endif
