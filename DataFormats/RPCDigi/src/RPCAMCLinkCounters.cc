#include "DataFormats/RPCDigi/interface/RPCAMCLinkCounters.h"

std::string RPCAMCLinkCounters::getTypeName(unsigned int type)
{
    switch (type) {
        // from FED CDF Header and Trailer
    case fed_event_:
        return std::string("Event"); break;
    case fed_header_check_fail_:
        return std::string("Header check fail"); break;
    case fed_header_id_mismatch_:
        return std::string("Header FED ID mismatch"); break;
    case fed_trailer_check_fail_:
        return std::string("Trailer check fail"); break;
    case fed_trailer_length_mismatch_:
        return std::string("Trailer length mismatch"); break;
    case fed_trailer_crc_mismatch_:
        return std::string("Trailer CRC mismatch"); break;
        // from FED Block Header
    case fed_block_length_invalid_:
        return std::string("Invalid block length"); break;
        // from FED Block Content
    case fed_block_amc_number_invalid_:
        return std::string("Invalid AMC number"); break;

    case amc_evc_bc_invalid_:
        return std::string("AMC EvC or BC invalid"); break;
    case amc_payload_length_invalid_:
        return std::string("Invalid payload length"); break;
        // from TwinMux Playload Header
    case amc_number_mismatch_:
        return std::string("AMC number mismatch"); break;

        // from RPC Record
    case amc_link_invalid_:
        return std::string("Invalid Link"); break;
    case amc_data_:
        return std::string("Data"); break;

    case input_data_:
        return std::string("Data"); break;
    case input_link_error_:
        return std::string("Link error"); break;
    case input_link_ack_fail_:
        return std::string("Link ack fail"); break;
    case input_eod_:
        return std::string("EOD"); break;
    case input_lb_invalid_:
        return std::string("Invalid LB"); break;
    case input_connector_invalid_:
        return std::string("Invalid Connector"); break;
    case input_connector_not_used_:
        return std::string("Connector not used"); break;
        /*
    case input_bc_mismatch_:
        return std::string("BC Mismatch"); break;
    case input_bc0_mismatch_:
        return std::string("BC0 Mismatch"); break;
        */

    default:
        return std::string("unknown"); break;
    }
}

RPCAMCLinkCounters::RPCAMCLinkCounters()
{}
