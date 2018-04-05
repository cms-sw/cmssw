#ifndef CondFormats_RPCObjects_RPCInverseLBLinkMap_h
#define CondFormats_RPCObjects_RPCInverseLBLinkMap_h

#include <cstdint>

#include <map>

#include "CondFormats/RPCObjects/interface/RPCLBLink.h"
#include "CondFormats/RPCObjects/interface/RPCFebConnector.h"

class RPCInverseLBLinkMap
{
public:
    typedef std::multimap<std::uint32_t, std::pair<RPCLBLink, RPCFebConnector> > map_type;

public:
    RPCInverseLBLinkMap();

    map_type & getMap();
    map_type const & getMap() const;

protected:
    map_type map_;
};

inline RPCInverseLBLinkMap::map_type & RPCInverseLBLinkMap::getMap()
{
    return map_;
}

inline RPCInverseLBLinkMap::map_type const & RPCInverseLBLinkMap::getMap() const
{
    return map_;
}

#endif // CondFormats_RPCObjects_RPCInverseLBLinkMap_h
