#ifndef TruncatedPyramid_h
#define TruncatedPyramid_h

#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include <CLHEP/Geometry/Point3D.h>
#include <CLHEP/Geometry/Plane3D.h>
#include <CLHEP/Geometry/Vector3D.h>
#include <CLHEP/Geometry/Transform3D.h>
#include <vector>
#include <cstdint>

/**

   \class TruncatedPyramid

   \brief A base class to handle the particular shape of Ecal Xtals. Taken from ORCA Calorimetry Code
   
*/

class TruncatedPyramid final : public CaloCellGeometry {
public:
  typedef CaloCellGeometry::CCGFloat CCGFloat;
  typedef CaloCellGeometry::Pt3D Pt3D;
  typedef CaloCellGeometry::Pt3DVec Pt3DVec;
  typedef CaloCellGeometry::Tr3D Tr3D;

  static constexpr uint32_t k_Dz = 0;     //Half-length along the z-axis
  static constexpr uint32_t k_Theta = 1;  //Polar angle of the line joining the
                                          //centres of the faces at -/+ Dz
  static constexpr uint32_t k_Phi = 2;    //Azimuthal angle of the line joing the
                                          //centres of the faces at -/+ Dz
  static constexpr uint32_t k_Dy1 = 3;    //Half-length along y of the face at -Dz
  static constexpr uint32_t k_Dx1 = 4;    //Half-length along x of the side at
                                          //y=-Dy1 of the face at -Dz
  static constexpr uint32_t k_Dx2 = 5;    //Half-length along x of the side at
                                          //y=+Dy1 of the face at -Dz
  static constexpr uint32_t k_Alp1 = 6;   //Angle w.r.t the y axis from the center
                                          //of the sides at y=-Dy1 to at y=+Dy1
  static constexpr uint32_t k_Dy2 = 7;    //Half-length along y of the face at +Dz
  static constexpr uint32_t k_Dx3 = 8;    //Half-length along x of the side at
                                          //y=-Dy2 of the face at +Dz
  static constexpr uint32_t k_Dx4 = 9;    //Half-length along x of the side at
                                          //y=+Dy2 of the face at +Dz
  static constexpr uint32_t k_Alp2 = 10;  //Angle w.r.t the y axis from the center
                                          //of the sides at y=-Dy2 to at y=+Dy2

  TruncatedPyramid(void);

  TruncatedPyramid(const TruncatedPyramid& tr);

  TruncatedPyramid& operator=(const TruncatedPyramid& tr);

  TruncatedPyramid(CornersMgr* cMgr,
                   const GlobalPoint& fCtr,
                   const GlobalPoint& bCtr,
                   const GlobalPoint& cor1,
                   const CCGFloat* parV);

  TruncatedPyramid(const CornersVec& corn, const CCGFloat* par);

  ~TruncatedPyramid() override;

  GlobalPoint getPosition(CCGFloat depth) const override;

  // Return thetaAxis polar angle of axis of the crystal
  CCGFloat getThetaAxis() const;

  // Return phiAxis azimuthal angle of axis of the crystal
  CCGFloat getPhiAxis() const;

  const GlobalVector& axis() const;

  // for geometry creation in other classes
  static void createCorners(const std::vector<CCGFloat>& pv, const Tr3D& tr, std::vector<GlobalPoint>& co);

  void vocalCorners(Pt3DVec& vec, const CCGFloat* pv, Pt3D& ref) const override;

  static void localCorners(Pt3DVec& vec, const CCGFloat* pv, Pt3D& ref);

  static void localCornersReflection(Pt3DVec& vec, const CCGFloat* pv, Pt3D& ref);

  static void localCornersSwap(Pt3DVec& vec, const CCGFloat* pv, Pt3D& ref);

  void getTransform(Tr3D& tr, Pt3DVec* lptr) const override;

private:
  void initCorners(CornersVec&) override;

  GlobalVector makeAxis(void);

  const GlobalPoint backCtr(void) const;
  GlobalVector m_axis;
  Pt3D m_corOne;
};

std::ostream& operator<<(std::ostream& s, const TruncatedPyramid& cell);

#endif
