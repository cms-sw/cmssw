#ifndef Geometry_RPCSimAlgo_RPCRollSpecs_H
#define Geometry_RPCSimAlgo_RPCRollSpecs_H

/** \class RPCRollSpecs
 *  Storage of the parameters of the RPC Chamber
 *  using standard topologies
 *
 * \author M. Maggi - INFN Bari
 *
 */
#include <vector>
#include <string>

class StripTopology;

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"


class RPCRollSpecs : public GeomDetType {

 public:
  typedef std::vector<float> RPCSpecs;

  RPCRollSpecs( SubDetector rss, const std::string& name, const RPCSpecs& pars);

  ~RPCRollSpecs();

  const Topology& topology() const;

  const StripTopology& specificTopology() const;

  const std::string& detName() const;

 private:
  StripTopology* _top;
  std::vector<float> _p;
  std::string _n;

};
#endif
