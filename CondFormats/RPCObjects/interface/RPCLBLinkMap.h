#ifndef CondFormats_RPCObjects_RPCLBLinkMap_h
#define CondFormats_RPCObjects_RPCLBLinkMap_h

#include <map>

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/RPCObjects/interface/RPCLBLink.h"
#include "CondFormats/RPCObjects/interface/RPCFebConnector.h"

class RPCLBLinkMap
{
public:
    typedef std::map<RPCLBLink, RPCFebConnector> map_type;

public:
    RPCLBLinkMap();

    map_type & getMap();
    map_type const & getMap() const;

protected:
    map_type map_;

    COND_SERIALIZABLE;
};

inline RPCLBLinkMap::map_type & RPCLBLinkMap::getMap()
{
    return map_;
}

inline RPCLBLinkMap::map_type const & RPCLBLinkMap::getMap() const
{
    return map_;
}

#endif // CondFormats_RPCObjects_RPCLBLinkMap_h
