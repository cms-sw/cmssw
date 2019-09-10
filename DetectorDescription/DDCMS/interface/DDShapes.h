#ifndef Detector_Description_DDCMS_DDShapes_h
#define Detector_Description_DDCMS_DDShapes_h

#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"

namespace cms {

  enum class DDSolidShape;

  namespace dd {

    DDSolidShape getCurrentShape(const cms::DDFilteredView &fview);

    class DDBox {
    public:
      DDBox(const cms::DDFilteredView &fview);
      DDBox(void) = delete;

      double halfX() const { return (dx_); }
      double halfY() const { return (dy_); }
      double halfZ() const { return (dz_); }

      const bool valid;  // Tells if DDBox contains valid values

    private:
      double dx_, dy_, dz_;
    };

    // A tube section
    class DDTubs {
    public:
      DDTubs(const cms::DDFilteredView &fview);
      DDTubs(void) = delete;

      double zhalf(void) const { return (zHalf_); }
      double rIn(void) const { return (rIn_); }
      double rOut(void) const { return (rOut_); }
      double startPhi(void) const { return (startPhi_); }
      double deltaPhi(void) const { return (deltaPhi_); }

      const bool valid;

    private:
      double zHalf_;
      double rIn_;
      double rOut_;
      double startPhi_;
      double deltaPhi_;
    };

    // A cone section
    class DDCons {
    public:
      DDCons(const cms::DDFilteredView &fview);
      DDCons(void) = delete;

      double zhalf(void) const { return (dz_); }
      double rInMinusZ(void) const { return (rmin1_); }
      double rOutMinusZ(void) const { return (rmax1_); }
      double rInPlusZ(void) const { return (rmin2_); }
      double rOutPlusZ(void) const { return (rmax2_); }
      double phiFrom(void) const { return (phi1_); }
      double deltaPhi(void) const { return (phi2_); }

      const bool valid;  // Tells if DDCons contains valid values

    private:
      double dz_;
      double rmin1_;
      double rmax1_;
      double rmin2_;
      double rmax2_;
      double phi1_;
      double phi2_;
    };

    // A simple trapezoid
    class DDTrap {
    public:
      DDTrap(const cms::DDFilteredView &fview);
      DDTrap(void) = delete;

      //! half of the z-Axis
      double halfZ(void) const { return (halfZ_); }

      //! Polar angle of the line joining the centres of the faces at -/+pDz
      double theta(void) const { return (theta_); }

      //! Azimuthal angle of the line joining the centres of the faces at -/+pDz
      double phi(void) const { return (phi_); }

      //! Half-length along y of the face at -pDz
      double y1(void) const { return (y1_); }

      //! Half-length along x of the side at y=-pDy1 of the face at -pDz
      double x1(void) const { return (x1_); }

      //! Half-length along x of the side at y=+pDy1 of the face at -pDz
      double x2(void) const { return (x2_); }

      //! Half-length along y of the face at +pDz
      double y2(void) const { return (y2_); }

      //! Half-length along x of the side at y=-pDy2 of the face at +pDz
      double x3(void) const { return (x3_); }

      //! Half-length along x of the side at y=+pDy2 of the face at +pDz
      double x4(void) const { return (x4_); }

      //! Angle with respect to the y axis from the centre of the side at y=-pDy1 to the centre at y=+pDy1 of the face at -pDz
      double alpha1(void) const { return (alpha1_); }

      //! Angle with respect to the y axis from the centre of the side at y=-pDy2 to the centre at y=+pDy2 of the face at +pDz
      double alpha2(void) const { return (alpha2_); }

      const bool valid;

    private:
      double halfZ_;
      double theta_;
      double phi_;
      double x1_;
      double x2_;
      double y1_;
      double y2_;
      double x3_;
      double x4_;
      double alpha1_;
      double alpha2_;
    };

    // A pseudo-trapezoid, that is, a trapezoid with a cylinder.
    // This shape is obsolete. This commented-out code can be removed after we are sure
    // it won't be needed with DD4hep.
    // The code does not work correctly.
    /*
    class DDPseudoTrap {
    public:
      DDPseudoTrap(const cms::DDFilteredView &fview);
      DDPseudoTrap(void) = delete;

      //! half of the z-Axis
      double halfZ(void) const { return (dz_); }
      //
      //! half length along x on -z
      double x1(void) const { return (minusX_); }

      //! half length along x on +z
      double x2(void) const { return (plusX_); }

      //! half length along y on -z
      double y1(void) const { return (minusY_); }

      //! half length along y on +z
      double y2(void) const { return (plusY_); }

      //! radius of the cut-out (neg.) or rounding (pos.)
      double radius(void) const { return (rmax_); }

      //! true, if cut-out or rounding is on the -z side
      bool atMinusZ(void) const { return (minusZSide_); }

      const bool valid;  // Tells if shape contains valid values

    private:
      double dz_;
      double minusX_;
      double plusX_;
      double minusY_;
      double plusY_;
      double rmax_;
      bool minusZSide_;
    };
    */

    /// A truncated tube section
    class DDTruncTubs {
    public:
      DDTruncTubs(const cms::DDFilteredView &fview);
      DDTruncTubs(void) = delete;

      //! half of the z-Axis
      double zHalf(void) const { return (zHalf_); }

      //! inner radius
      double rIn(void) const { return (rIn_); }

      //! outer radius
      double rOut(void) const { return (rOut_); }

      //! angular start of the tube-section
      double startPhi(void) const { return (startPhi_); }

      //! angular span of the tube-section
      double deltaPhi(void) const { return (deltaPhi_); }

      //! truncation at begin of the tube-section
      double cutAtStart(void) const { return (cutAtStart_); }

      //! truncation at end of the tube-section
      double cutAtDelta(void) const { return (cutAtDelta_); }

      //! true, if truncation is on the inner side of the tube-section
      bool cutInside(void) const { return (cutInside_); }

      const bool valid;

    private:
      double zHalf_;
      double rIn_;
      double rOut_;
      double startPhi_;
      double deltaPhi_;
      double cutAtStart_;
      double cutAtDelta_;
      bool cutInside_;
    };
  }  // namespace dd
}  // namespace cms

#endif
