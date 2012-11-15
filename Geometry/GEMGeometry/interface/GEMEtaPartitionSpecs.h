#ifndef Geometry_GEMGeometry_GEMEtaPartitionSpecs_H
#define Geometry_GEMGeometry_GEMEtaPartitionSpecs_H

/** \class GEMEtaPartitionSpecs
 *  Storage of the parameters of the GEM Chamber
 *  using standard topologies
 *
 * \author M. Maggi - INFN Bari
 *
 */
#include <vector>
#include <string>

class StripTopology;

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"


class GEMEtaPartitionSpecs : public GeomDetType {

 public:
  typedef std::vector<float> GEMSpecs;

  GEMEtaPartitionSpecs( SubDetector rss, const std::string& name, const GEMSpecs& pars);

  ~GEMEtaPartitionSpecs();

  const Topology& topology() const;

  const StripTopology& specificTopology() const;

  const std::string& detName() const;

 private:
  StripTopology* _top;
  std::vector<float> _p;
  std::string _n;

};
#endif
