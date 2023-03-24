#include <Geometry/CommonTopologies/interface/TkRadialStripTopology.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <iostream>
#include <cmath>
#include <algorithm>
#include <cassert>

#include <vdt/vdtMath.h>

#ifdef MATH_STS
#include <iostream>
#endif
namespace {

#ifdef MATH_STS
  struct Stat {
    Stat(const char* in) : name(in){};
    ~Stat() {
      edm::LogVerbatim("CommonTopologies") << name << ": atan0 calls tot/large/over1: " << natan << "/" << nlarge << "/" << over1;
    }

    void add(float t) {
      auto at = std::abs(t);
      ++natan;
      if (at > 0.40f)
        ++nlarge;
      if (at > 1.0)
        ++over1;
    }
    const char* name;
    long long natan = 0;
    long long nlarge = 0;
    long long over1 = 0;
  };

  Stat statM("mpos");
  Stat statS("span");
#endif

  // valid for |x| < 0.15  (better then 10^-9
  template <typename T>
  inline T tan15(T x) {
    return x * (T(1) + (x * x) * (T(0.33331906795501708984375) + (x * x) * T(0.135160386562347412109375)));
  }

  // valid for z < pi/8
  //  x * (1 + x*x * (-0.33322894573211669921875 + x*x * (0.1967026889324188232421875 + x*x * (-0.11053790152072906494140625))))  // .1e-7 by Sollya
  inline float atan0(float t) {
    auto z = t;
    // if( t > 0.4142135623730950f ) // * tan pi/8
    // z = (t-1.0f)/(t+1.0f);
    float z2 = z * z;
    float ret =
        (((8.05374449538e-2f * z2 - 1.38776856032E-1f) * z2 + 1.99777106478E-1f) * z2 - 3.33329491539E-1f) * z2 * z + z;
    // if( t > 0.4142135623730950f ) ret +=0.7853981633974483096f;
    return ret;
  }

  inline float atanClip(float t) {
    constexpr float tanPi8 = 0.4142135623730950;
    constexpr float pio8 = 3.141592653589793238 / 8;
    float at = std::abs(t);
    return std::copysign((at < tanPi8) ? atan0(at) : pio8, t);
  }

}  // namespace

TkRadialStripTopology::TkRadialStripTopology(int ns, float aw, float dh, float r, int yAx, float yMid)
    : theNumberOfStrips(ns),
      theAngularWidth(aw),
      theAWidthInverse(1.f / aw),
      theTanAW(std::tan(aw)),
      theDetHeight(dh),
      theCentreToIntersection(r),
      theYAxisOrientation(yAx),
      yCentre(yMid),
      theRadialSigma(std::pow(dh, 2.f) * (1.f / 12.f)) {
  // Angular offset of extreme edge of detector, so that angle is
  // zero for a strip lying along local y axis = long symmetry axis of plane of strips
  thePhiOfOneEdge = -(0.5 * theNumberOfStrips) * theAngularWidth;  // always negative!
  theTanOfOneEdge = std::tan(std::abs(thePhiOfOneEdge));
  assert(std::abs(thePhiOfOneEdge) < 0.15);  //

  LogTrace("TkRadialStripTopology") << "TkRadialStripTopology: constructed with"
                                    << " strips = " << ns << " width = " << aw << " rad "
                                    << " det_height = " << dh << " ctoi = " << r << " phi_edge = " << thePhiOfOneEdge
                                    << " rad "
                                    << " y_ax_ori = " << theYAxisOrientation << " y_det_centre = " << yCentre << "\n";
}

int TkRadialStripTopology::channel(const LocalPoint& lp) const {
  return std::min(int(strip(lp)), theNumberOfStrips - 1);
}

int TkRadialStripTopology::nearestStrip(const LocalPoint& lp) const {
  return std::min(nstrips(), static_cast<int>(std::max(float(0), strip(lp))) + 1);
}

float TkRadialStripTopology::yDistanceToIntersection(float y) const {
  return yAxisOrientation() * y + originToIntersection();
}

float TkRadialStripTopology::localStripLength(const LocalPoint& lp) const {
  return detHeight() * std::sqrt(1.f + std::pow(lp.x() / yDistanceToIntersection(lp.y()), 2.f));
}

float TkRadialStripTopology::xOfStrip(int strip, float y) const {
  return yAxisOrientation() * yDistanceToIntersection(y) * std::tan(stripAngle(static_cast<float>(strip) - 0.5f));
}

float TkRadialStripTopology::strip(const LocalPoint& lp) const {
  // phi is measured from y axis --> sign of angle is sign of x * yAxisOrientation --> use atan2(x,y), not atan2(y,x)
  const float phi = atanClip(lp.x() / yDistanceToIntersection(lp.y()));
  const float aStrip = (phi - phiOfOneEdge()) * theAWidthInverse;
  return std::max(float(0), std::min((float)nstrips(), aStrip));
}

float TkRadialStripTopology::coveredStrips(const LocalPoint& lp1, const LocalPoint& lp2) const {
  // http://en.wikipedia.org/wiki/List_of_trigonometric_identities#Angle_sum_and_difference_identities
  // atan(a)-atan(b) = atan( (a-b)/(1+a*b) )
  // avoid divisions
  // float t1 = lp1.x()/yDistanceToIntersection( lp1.y() );
  // float t2 = lp2.x()/yDistanceToIntersection( lp2.y() );
  // float t = (t1-t2)/(1.+t1*t2);
  auto y1 = yDistanceToIntersection(lp1.y());
  auto y2 = yDistanceToIntersection(lp2.y());
  auto x1 = lp1.x();
  auto x2 = lp2.x();

  auto t = (y2 * x1 - y1 * x2) / (y1 * y2 + x1 * x2);

#ifdef MATH_STS
  statS.add(t);
#endif
  //  edm::LogVerbatim("CommonTopologies") << "atans " << atanClip(t) << " " << std::atan2(lp1.x(), yDistanceToIntersection(lp1.y())) - std::atan2(lp2.x(),yDistanceToIntersection(lp2.y()));
  // clip???
  return atanClip(t) * theAWidthInverse;
  //   return (measurementPosition(lp1)-measurementPosition(lp2)).x();
}

LocalPoint TkRadialStripTopology::localPosition(float strip) const {
  return LocalPoint(yAxisOrientation() * originToIntersection() * tan15(stripAngle(strip)), 0);
}

LocalPoint TkRadialStripTopology::localPosition(const MeasurementPoint& mp) const {
  const float  // y = (L/cos(phi))*mp.y()*cos(phi)
      y(mp.y() * detHeight() + yCentreOfStripPlane()),
      x(yAxisOrientation() * yDistanceToIntersection(y) * tan15(stripAngle(mp.x())));
  return LocalPoint(x, y);
}

MeasurementPoint TkRadialStripTopology::measurementPosition(const LocalPoint& lp) const {
  // phi is [pi/2 - conventional local phi], use atan2(x,y) rather than atan2(y,x)
  // clip   ( at pi/8 or detedge+tollerance?)
  float t = lp.x() / yDistanceToIntersection(lp.y());
#ifdef MATH_STS
  statM.add(t);
#endif
  const float phi = atanClip(t);
  return MeasurementPoint((phi - phiOfOneEdge()) * theAWidthInverse, (lp.y() - yCentreOfStripPlane()) / detHeight());
}

LocalError TkRadialStripTopology::localError(float strip, float stripErr2) const {
  double phi = stripAngle(strip);

  const double t1(tan15(phi)),  // std::tan(phif)), // (vdt::fast_tanf(phif)),
      t2(t1 * t1),
      // s1(std::sin(phi)), c1(std::cos(phi)),
      // cs(s1*c1), s2(s1*s1), c2(1-s2), // rotation matrix

      tt(stripErr2 * std::pow(centreToIntersection() * angularWidth(), 2.f)),  // tangential sigma^2   *c2
      rr(theRadialSigma),  // radial sigma^2( uniform prob density along strip)  *c2

      xx(tt + t2 * rr), yy(t2 * tt + rr), xy(t1 * (rr - tt));

  return LocalError(xx, xy, yy);
}

LocalError TkRadialStripTopology::localError(const MeasurementPoint& mp, const MeasurementError& me) const {
  const double phi(stripAngle(mp.x())), s1(std::sin(phi)), c1(std::cos(phi)), cs(s1 * c1), s2(s1 * s1),
      c2(1 - s2),  // rotation matrix

      T(angularWidth() * (centreToIntersection() + yAxisOrientation() * mp.y() * detHeight()) /
        c1),                // tangential measurement unit (local pitch)
      R(detHeight() / c1),  // radial measurement unit (strip length)
      tt(me.uu() * T * T),  // tangential sigma^2
      rr(me.vv() * R * R),  // radial sigma^2
      tr(me.uv() * T * R),

      xx(c2 * tt + 2 * cs * tr + s2 * rr), yy(s2 * tt - 2 * cs * tr + c2 * rr), xy(cs * (rr - tt) + tr * (c2 - s2));

  return LocalError(xx, xy, yy);
}

MeasurementError TkRadialStripTopology::measurementError(const LocalPoint& p, const LocalError& e) const {
  const double yHitToInter(yDistanceToIntersection(p.y())),
      t(yAxisOrientation() * p.x() / yHitToInter),  // tan(strip angle)
      cs(t / (1 + t * t)), s2(t * cs), c2(1 - s2),  // rotation matrix

      T2(1. / (std::pow(angularWidth(), 2.f) *
               (std::pow(p.x(), 2.f) + std::pow(yHitToInter, 2)))),  // 1./tangential measurement unit (local pitch) ^2
      R2(c2 / std::pow(detHeight(), 2.f)),                           // 1./ radial measurement unit (strip length) ^2

      uu((c2 * e.xx() - 2 * cs * e.xy() + s2 * e.yy()) * T2), vv((s2 * e.xx() + 2 * cs * e.xy() + c2 * e.yy()) * R2),
      uv((cs * (e.xx() - e.yy()) + e.xy() * (c2 - s2)) * std::sqrt(T2 * R2));

  return MeasurementError(uu, uv, vv);
}

// The local pitch is the local x width of the strip at the local (x,y)
float TkRadialStripTopology::localPitch(const LocalPoint& lp) const {
  // this should be ~ y*(tan(phi+aw)-tan(phi)) = -x + y*(tan(aw)+tan(phi))/(1.f-tan(aw)*tan(phi)) tan(phi)=x/y
  float y = yDistanceToIntersection(lp.y());
  float x = std::abs(lp.x());
  return y * (y * theTanAW + x) / (y - theTanAW * x) - x;
}

/* old version
float TkRadialStripTopology::localPitch(const LocalPoint& lp) const { 
  // this should be ~ y*(tan(phi+aw)-tan(phi)) = -tan(phi) + (tan(aw)+tan(phi))/(1.f-tan(aw)*tan(phi)) 
  const int istrip = std::min(nstrips(), static_cast<int>(strip(lp)) + 1); // which strip number
  float fangle = stripAngle(static_cast<float>(istrip) - 0.5); // angle of strip centre
  float p =
    yDistanceToIntersection( lp.y() ) * std::sin(angularWidth()) /
    std::pow( std::cos(fangle-0.5f*angularWidth()), 2.f);

  float theTanAW = std::tan(theAngularWidth);
  float y =  yDistanceToIntersection( lp.y() );
  float x = std::abs(lp.x());
  float myP = y*(y*theTanAW+x)/(y-theTanAW*x)-x; // (y*theTanAW+x)/(1.f-theTanAW*x/y)-x;
  edm::LogVerbatim("CommonTopologies") << "localPitch " << p << " " << myP;

  return p;

}
*/
