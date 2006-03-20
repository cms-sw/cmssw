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

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"


class RPCRollSpecs : public GeomDetType {

 public:
  typedef std::vector<float> RPCSpecs;

  RPCRollSpecs( SubDetector rss, const RPCSpecs& pars);
  ~RPCRollSpecs();

  const Topology& topology() const;
  const GeomDetType& type() const;
 private:
  Topology* _top;
  std::vector<float> _p;

};
#endif
