#ifndef DataFormats_RPCDigi_RPCAMCLinkCounters_h
#define DataFormats_RPCDigi_RPCAMCLinkCounters_h

#include <cstdint>
#include <map>
#include <string>

#include "CondFormats/RPCObjects/interface/RPCAMCLink.h"

class RPCAMCLinkCounters
{
public:
    typedef std::map<std::pair<unsigned int, std::uint32_t>, unsigned int > map_type;
    typedef map_type::iterator          iterator;
    typedef map_type::const_iterator    const_iterator;

    // from FED CDF Header and Trailer
    static unsigned int const fed_event_                    = 0;
    static unsigned int const fed_header_check_fail_        = 1;
    static unsigned int const fed_header_id_mismatch_       = 2;
    static unsigned int const fed_trailer_check_fail_       = 3;
    static unsigned int const fed_trailer_length_mismatch_  = 4;
    static unsigned int const fed_trailer_crc_mismatch_     = 5;
    // from FED Block Header
    static unsigned int const fed_block_length_invalid_     = 6;
    // from FED Block Content
    static unsigned int const fed_block_amc_number_invalid_ = 7;

    static unsigned int const amc_evc_bc_invalid_           = 101;
    static unsigned int const amc_payload_length_invalid_   = 102;
    // from TwinMux Playload Header
    static unsigned int const amc_number_mismatch_          = 103;

    // from RPC Record
    static unsigned int const amc_link_invalid_             = 104;
    static unsigned int const amc_data_                     = 100;

    static unsigned int const input_data_                   = 200;
    static unsigned int const input_link_error_             = 201;
    static unsigned int const input_link_ack_fail_          = 202;
    static unsigned int const input_eod_                    = 203;
    static unsigned int const input_lb_invalid_             = 204;
    static unsigned int const input_connector_invalid_      = 205;
    static unsigned int const input_connector_not_used_     = 206;
    // static unsigned int const input_bc_mismatch_            = 207;
    // static unsigned int const input_bc0_mismatch_           = 208;

    static unsigned int const fed_min_                      = 0;
    static unsigned int const fed_max_                      = 7;
    static unsigned int const amc_min_                      = 100;
    static unsigned int const amc_max_                      = 104;
    static unsigned int const input_min_                    = 200;
    static unsigned int const input_max_                    = 206;

    static std::string getTypeName(unsigned int _type);

public:
    RPCAMCLinkCounters();

    void add(unsigned int _type, RPCAMCLink const & _link, unsigned int _count = 1);
    void reset();
    void reset(unsigned int _type);
    void reset(unsigned int _type, RPCAMCLink const & _link);

    std::pair<const_iterator, const_iterator> getCounters() const;
    std::pair<const_iterator, const_iterator> getCounters(unsigned int _type) const;
    std::pair<const_iterator, const_iterator> getCounters(unsigned int _lower_type, unsigned int _upper_type) const;

protected:
    map_type type_link_count_;
};

#include "DataFormats/RPCDigi/interface/RPCAMCLinkCounters.icc"

#endif // DataFormats_RPCDigi_RPCAMCLinkCounters_h
