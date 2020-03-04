/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#include "RecoCTPPS/TotemRPLocal/interface/FastLineRecognition.h"

#include "DataFormats/CTPPSReco/interface/TotemRPRecHit.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"

#include <map>
#include <cmath>
#include <cstdio>
#include <algorithm>

//#define CTPPS_DEBUG 1

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

const double FastLineRecognition::sigma0 = 66E-3 / sqrt(12.);

//----------------------------------------------------------------------------------------------------

void FastLineRecognition::Cluster::add(const Point *p1, const Point *p2, double a, double b, double w) {
  // which points to be added to contents?
  bool add1 = true, add2 = true;
  for (vector<const Point *>::const_iterator it = contents.begin(); it != contents.end() && (add1 || add2); ++it) {
    if ((*it)->hit == p1->hit)
      add1 = false;

    if ((*it)->hit == p2->hit)
      add2 = false;
  }

  // add the points
  if (add1)
    contents.push_back(p1);
  if (add2)
    contents.push_back(p2);

  // update sums
  Saw += a * w;
  Sbw += b * w;
  Sw += w;
  S1 += 1.;
}

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

FastLineRecognition::FastLineRecognition(double cw_a, double cw_b)
    : chw_a(cw_a / 2.), chw_b(cw_b / 2.), geometry(nullptr) {}

//----------------------------------------------------------------------------------------------------

FastLineRecognition::~FastLineRecognition() {}

//----------------------------------------------------------------------------------------------------

FastLineRecognition::GeomData FastLineRecognition::getGeomData(unsigned int id) {
  // result already buffered?
  map<unsigned int, GeomData>::iterator it = geometryMap.find(id);
  if (it != geometryMap.end())
    return it->second;

  // calculate it
  const auto &d = geometry->localToGlobalDirection(id, CTPPSGeometry::Vector(0., 1., 0.));
  DetGeomDesc::Translation c = geometry->sensor(TotemRPDetId(id))->translation();
  GeomData gd;
  gd.z = c.z();
  gd.s = d.x() * c.x() + d.y() * c.y();

  geometryMap[id] = gd;

  return gd;
}

//----------------------------------------------------------------------------------------------------

void FastLineRecognition::getPatterns(const DetSetVector<TotemRPRecHit> &input,
                                      double z0,
                                      double threshold,
                                      DetSet<TotemRPUVPattern> &patterns) {
  // build collection of points in the global coordinate system
  std::vector<Point> points;
  for (auto &ds : input) {
    unsigned int detId = ds.detId();

    for (auto &h : ds) {
      const TotemRPRecHit *hit = &h;
      const GeomData &gd = getGeomData(detId);

      double p = hit->position() + gd.s;
      double z = gd.z - z0;
      double w = sigma0 / hit->sigma();

      points.push_back(Point(detId, hit, p, z, w));
    }
  }

#if CTPPS_DEBUG > 0
  printf(">> FastLineRecognition::getPatterns(z0 = %E)\n", z0);
  printf(">>>>>>>>>>>>>>>>>>>\n");
#endif

  // reset output
  patterns.clear();

  Cluster c;
  while (getOneLine(points, threshold, c)) {
    // convert cluster to pattern and save it
    TotemRPUVPattern pattern;
    pattern.setA(c.Saw / c.Sw);
    pattern.setB(c.Sbw / c.Sw);
    pattern.setW(c.weight);

#if CTPPS_DEBUG > 0
    printf("\tpoints of the selected cluster: %lu\n", c.contents.size());
#endif

    for (auto &pit : c.contents) {
#if CTPPS_DEBUG > 0
      printf("\t\t%.1f\n", pit->z);
#endif
      pattern.addHit(pit->detId, *(pit->hit));
    }

    patterns.push_back(pattern);

#if CTPPS_DEBUG > 0
    unsigned int u_points_b = 0;
    for (vector<Point>::iterator dit = points.begin(); dit != points.end(); ++dit)
      if (dit->usable)
        u_points_b++;
    printf("\tusable points before: %u\n", u_points_b);
#endif

    // remove points belonging to the recognized line
    for (vector<const Point *>::iterator hit = c.contents.begin(); hit != c.contents.end(); ++hit) {
      for (vector<Point>::iterator dit = points.begin(); dit != points.end(); ++dit) {
        //printf("\t\t1: %.2f, %p vs. 2: %.2f, %p\n", (*hit)->z, (*hit)->hit, dit->z, dit->hit);
        if ((*hit)->hit == dit->hit) {
          dit->usable = false;
          //points.erase(dit);
          break;
        }
      }
    }

#if CTPPS_DEBUG > 0
    unsigned int u_points_a = 0;
    for (vector<Point>::iterator dit = points.begin(); dit != points.end(); ++dit)
      if (dit->usable)
        u_points_a++;
    printf("\tusable points after: %u\n", u_points_a);
#endif
  }

#if CTPPS_DEBUG > 0
  printf("patterns at end: %lu\n", patterns.size());
  printf("<<<<<<<<<<<<<<<<<<<\n");
#endif
}

//----------------------------------------------------------------------------------------------------

bool FastLineRecognition::getOneLine(const vector<FastLineRecognition::Point> &points,
                                     double threshold,
                                     FastLineRecognition::Cluster &result) {
#if CTPPS_DEBUG > 0
  printf("\tFastLineRecognition::getOneLine\n");
#endif

  if (points.size() < 2)
    return false;

  vector<Cluster> clusters;

  // go through all the combinations of measured points
  for (vector<Point>::const_iterator it1 = points.begin(); it1 != points.end(); ++it1) {
    if (!it1->usable)
      continue;

    for (vector<Point>::const_iterator it2 = it1; it2 != points.end(); ++it2) {
      if (!it2->usable)
        continue;

      const double &z1 = it1->z;
      const double &z2 = it2->z;

      if (z1 == z2)
        continue;

      const double &p1 = it1->h;
      const double &p2 = it2->h;

      const double &w1 = it1->w;
      const double &w2 = it2->w;

      // calculate intersection
      double a = (p2 - p1) / (z2 - z1);
      double b = p1 - z1 * a;
      double w = w1 + w2;

#if CTPPS_DEBUG > 0
      printf("\t\t\tz: 1=%+5.1f, 2=%+5.1f | U/V: 1=%+6.3f, 2=%+6.3f | a=%+6.3f rad, b=%+6.3f mm, w=%.1f\n",
             z1,
             z2,
             p1,
             p2,
             a,
             b,
             w);
#endif

      // add it to the appropriate cluster
      bool newCluster = true;
      for (unsigned int k = 0; k < clusters.size(); k++) {
        Cluster &c = clusters[k];
        if (c.S1 < 1. || c.Sw <= 0.)
          continue;

#if CTPPS_DEBUG > 0
        if (k < 10)
          printf("\t\t\t\ttest cluster %u at a=%+6.3f, b=%+6.3f : %+6.3f, %+6.3f : %i, %i\n",
                 k,
                 c.Saw / c.Sw,
                 c.Sbw / c.Sw,
                 chw_a,
                 chw_b,
                 (std::abs(a - c.Saw / c.Sw) < chw_a),
                 (std::abs(b - c.Sbw / c.Sw) < chw_b));
#endif

        if ((std::abs(a - c.Saw / c.Sw) < chw_a) && (std::abs(b - c.Sbw / c.Sw) < chw_b)) {
          newCluster = false;
          clusters[k].add(&(*it1), &(*it2), a, b, w);
#if CTPPS_DEBUG > 0
          printf("\t\t\t\t--> cluster %u\n", k);
#endif
          break;
        }
      }

      // make new cluster
      if (newCluster) {
#if CTPPS_DEBUG > 0
        printf("\t\t\t\t--> new cluster %lu\n", clusters.size());
#endif
        clusters.push_back(Cluster());
        clusters.back().add(&(*it1), &(*it2), a, b, w);
      }
    }
  }

#if CTPPS_DEBUG > 0
  printf("\t\tclusters: %lu\n", clusters.size());
#endif

  // find the cluster with highest weight
  unsigned int mk = 0;
  double mw = -1.;
  for (unsigned int k = 0; k < clusters.size(); k++) {
    double w = 0;
    for (vector<const Point *>::iterator it = clusters[k].contents.begin(); it != clusters[k].contents.end(); ++it)
      w += (*it)->w;
    clusters[k].weight = w;

    if (w > mw) {
      mw = w;
      mk = k;
    }
  }

#if CTPPS_DEBUG > 0
  printf("\t\tmw = %.1f, mk = %u\n", mw, mk);
#endif

  // rerturn result
  if (mw >= threshold) {
    result = clusters[mk];

    return true;
  } else
    return false;
}
