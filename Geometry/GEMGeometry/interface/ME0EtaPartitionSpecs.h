#ifndef Geometry_ME0Geometry_ME0EtaPartitionSpecs_H
#define Geometry_ME0Geometry_ME0EtaPartitionSpecs_H

/** \class ME0EtaPartitionSpecs
 *  Storage of the parameters of the ME0 Chamber
 *  using standard topologies
 *
 * \author M. Maggi - INFN Bari
 *
 */
#include <vector>
#include <string>

class StripTopology;

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"


class ME0EtaPartitionSpecs : public GeomDetType
{
public:

  typedef std::vector<float> ME0Specs;

  ME0EtaPartitionSpecs( SubDetector rss, const std::string& name, const ME0Specs& pars);

  ~ME0EtaPartitionSpecs();

  const Topology& topology() const;

  const StripTopology& specificTopology() const;

  const Topology& padTopology() const;

  const StripTopology& specificPadTopology() const;

  const std::string& detName() const;

  const ME0Specs& parameters() const;

private:
  
  /// topology of strips
  StripTopology* _top;

  /// topology of trigger pads (pad = bundle of strips, basically, a "fat" strip)
  StripTopology* _top_pad;

  std::vector<float> _p;
  std::string _n;
};
#endif
