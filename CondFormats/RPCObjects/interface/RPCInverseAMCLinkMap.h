#ifndef CondFormats_RPCObjects_RPCInverseAMCLinkMap_h
#define CondFormats_RPCObjects_RPCInverseAMCLinkMap_h

#include <map>

#include "CondFormats/RPCObjects/interface/RPCAMCLink.h"
#include "CondFormats/RPCObjects/interface/RPCLBLink.h"

class RPCInverseAMCLinkMap {
public:
  typedef std::multimap<RPCLBLink, RPCAMCLink> map_type;

public:
  RPCInverseAMCLinkMap();

  map_type& getMap();
  map_type const& getMap() const;

protected:
  map_type map_;
};

inline RPCInverseAMCLinkMap::map_type& RPCInverseAMCLinkMap::getMap() { return map_; }

inline RPCInverseAMCLinkMap::map_type const& RPCInverseAMCLinkMap::getMap() const { return map_; }

#endif  // CondFormats_RPCObjects_RPCInverseAMCLinkMap_h
