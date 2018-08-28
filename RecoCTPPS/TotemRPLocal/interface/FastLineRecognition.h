/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*   Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#ifndef RecoCTPPS_TotemRPLocal_FastLineRecognition
#define RecoCTPPS_TotemRPLocal_FastLineRecognition

#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"
#include "DataFormats/CTPPSReco/interface/TotemRPRecHit.h"
#include "DataFormats/CTPPSReco/interface/TotemRPUVPattern.h"


/**
 * \brief Class performing optimized hough transform to recognize lines.
**/

class FastLineRecognition
{
  public:
    FastLineRecognition(double cw_a = 0., double cw_b = 0.);

    ~FastLineRecognition();

    void resetGeometry(const CTPPSGeometry *_g)
    {
      geometry = _g;
      geometryMap.clear();
    }

    void getPatterns(const edm::DetSetVector<TotemRPRecHit> &input, double _z0, double threshold,
      edm::DetSet<TotemRPUVPattern> &patterns);

  private:
    /// the uncertainty of 1-hit cluster, in mm
    static const double sigma0;

    /// "typical" z
    double z0;

    /// cluster half widths in a and b
    double chw_a, chw_b;

    /// weight threshold for accepting pattern candidates (clusters)
    double threshold;

    /// pointer to the geometry
    const CTPPSGeometry* geometry;

    struct GeomData
    {
      double z;   ///< z position of a sensor (wrt. IP)
      double s;   ///< sensor's centre projected to its read-out direction
    };

    /// map: raw detector id --> GeomData
    std::map<unsigned int, GeomData> geometryMap;

    /// expects raw detector id
    GeomData getGeomData(unsigned int id);

    struct Point
    {
      unsigned int detId;       ///< raw detector id
      const TotemRPRecHit* hit; ///< pointer to original reco hit
      double h;                 ///< hit position in global coordinate system
      double z;                 ///< z position with respect to z0
      double w;                 ///< weight
      bool usable;              ///< whether the point can still be used
      Point(unsigned int _d=0, const TotemRPRecHit* _hit=nullptr, double _h=0., double _z=0., double _w=0.) :
        detId(_d), hit(_hit), h(_h), z(_z), w(_w), usable(true) {}
    };
    
    /// cluster of intersection points
    struct Cluster
    {
      double Saw, Sbw, Sw, S1;
      double weight;

      std::vector<const Point *> contents;

      Cluster() : Saw(0.), Sbw(0.), Sw(0.), S1(0.), weight(0.) {}

      void add(const Point *p1, const Point *p2, double a, double b, double w);

      bool operator<(const Cluster &c) const
      {
        return (this->Sw > c.Sw) ? true : false;
      }
    };

    /// gets the most significant pattern in the (remaining) points
    /// returns true when a pattern was found
    bool getOneLine(const std::vector<Point> &points, double threshold, Cluster &result);
};

#endif

