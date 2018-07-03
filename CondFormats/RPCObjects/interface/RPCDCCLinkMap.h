#ifndef CondFormats_RPCObjects_RPCDCCLinkMap_h
#define CondFormats_RPCObjects_RPCDCCLinkMap_h

#include <map>

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/RPCObjects/interface/RPCDCCLink.h"
#include "CondFormats/RPCObjects/interface/RPCLBLink.h"

class RPCDCCLinkMap
{
public:
    typedef std::map<RPCDCCLink, RPCLBLink> map_type;

public:
    RPCDCCLinkMap();

    map_type & getMap();
    map_type const & getMap() const;

protected:
    map_type map_;

    COND_SERIALIZABLE;
};

inline RPCDCCLinkMap::map_type & RPCDCCLinkMap::getMap()
{
    return map_;
}

inline RPCDCCLinkMap::map_type const & RPCDCCLinkMap::getMap() const
{
    return map_;
}

#endif // CondFormats_RPCObjects_RPCDCCLinkMap_h
