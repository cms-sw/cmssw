#ifndef FastSimDataFormats_CTPPSFastSim_CTPPSFastTrack_H
#define FastSimDataFormats_CTPPSFastSim_CTPPSFastTrack_H

#include "DataFormats/Math/interface/Vector.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"

#include <vector>
class CTPPSFastTrack {
public:
  typedef math::XYZVector Vector;
  typedef math::XYZPoint Point;
  // ~CTPPSFastTrack() {}
  CTPPSFastTrack()
      : thet(0.),
        thexi(0.),
        thecellid(0),
        thetof(0.),
        thex1(0.),
        they1(0.),
        thex2(0.),
        they2(0.),
        momentum_(0, 0, 0),
        vertex_(0, 0, 0) {}
  // constructor
  CTPPSFastTrack(float t,
                 float xi,
                 unsigned int cellid,
                 float tof,
                 float x1,
                 float y1,
                 float x2,
                 float y2,
                 const Vector &momentum,
                 const Point &vertex)
      : thet(t),
        thexi(xi),
        thecellid(cellid),
        thetof(tof),
        thex1(x1),
        they1(y1),
        thex2(x2),
        they2(y2),
        momentum_(momentum),
        vertex_(vertex) {}

  ////////////////////////////
  //
  /// track momentum vector
  const Vector &momentum() const;
  /// Reference point on the track
  const Point &referencePoint() const;
  // reference point on the track. This method is DEPRECATED, please use referencePoint() instead
  const Point &vertex() const;
  /* Time of flight in nanoseconds from the primary interaction
         *  to the entry point. Always positive in a PSimHit,
         *  but may become negative in a SimHit due to bunch assignment.
         */
  float timeOfFlight() const { return tof(); }

  float t() const { return thet; }

  float xi() const { return thexi; }

  float tof() const { return thetof; }

  float x1() const { return thex1; }

  float y1() const { return they1; }

  float x2() const { return thex2; }

  float y2() const { return they2; }
  float px() const { return momentum_.x(); }
  float py() const { return momentum_.Y(); }
  float pz() const { return momentum_.Z(); }
  float x0() const { return vertex_.x(); }
  float y0() const { return vertex_.Y(); }
  float z0() const { return vertex_.Z(); }

  unsigned int cellid() const { return thecellid; }

  void setp(const Vector &momentum) { momentum_ = momentum; }

  void setvertex(const Point &vertex) { vertex_ = vertex; }

  void settof(float tof) { thetof = tof; }

  void sett(float t) { thet = t; }

  void setxi(float xi) { thexi = xi; }

  void setx1(float x1) { thex1 = x1; }

  void sety1(float y1) { they1 = y1; }

  void setx2(float x2) { thex2 = x2; }

  void sety2(float y2) { they2 = y2; }

  void setcellid(unsigned int cellid) { thecellid = cellid; }

private:
  float thet;
  float thexi;
  unsigned int thecellid;
  float thetof;
  float thex1;
  float they1;
  float thex2;
  float they2;
  Vector momentum_;
  Point vertex_;
};

#endif  //CTPPSFastTrack_H
