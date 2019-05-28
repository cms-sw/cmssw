#ifndef CondFormats_RPCObjects_RPCAMCLinkMap_h
#define CondFormats_RPCObjects_RPCAMCLinkMap_h

#include <map>

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/RPCObjects/interface/RPCAMCLink.h"
#include "CondFormats/RPCObjects/interface/RPCLBLink.h"

class RPCAMCLinkMap {
public:
  typedef std::map<RPCAMCLink, RPCLBLink> map_type;

public:
  RPCAMCLinkMap();

  map_type& getMap();
  map_type const& getMap() const;

protected:
  map_type map_;

  COND_SERIALIZABLE;
};

inline RPCAMCLinkMap::map_type& RPCAMCLinkMap::getMap() { return map_; }

inline RPCAMCLinkMap::map_type const& RPCAMCLinkMap::getMap() const { return map_; }

#endif  // CondFormats_RPCObjects_RPCAMCLinkMap_h
