#ifndef DataFormats_RPCDigi_RPCAMCLinkCounters_h
#define DataFormats_RPCDigi_RPCAMCLinkCounters_h

#include <cstdint>
#include <map>

#include "CondFormats/RPCObjects/interface/RPCAMCLink.h"

class RPCAMCLinkCounters
{
public:
    typedef std::map<std::pair<unsigned int, std::uint32_t>, unsigned int> map_type;

public:
    RPCAMCLinkCounters();

    void add(unsigned int event, RPCAMCLink const & link, unsigned int count = 1);
    void reset();
    void reset(unsigned int event);
    void reset(unsigned int event, RPCAMCLink const & link);

    map_type const & getCounters() const;

protected:
    map_type type_link_count_;
};

#include "DataFormats/RPCDigi/interface/RPCAMCLinkCounters.icc"

#endif // DataFormats_RPCDigi_RPCAMCLinkCounters_h
