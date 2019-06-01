#ifndef EventFilter_RPCRawToDigi_RPCAMCLinkEvents_h
#define EventFilter_RPCRawToDigi_RPCAMCLinkEvents_h

#include <string>

#include "EventFilter/RPCRawToDigi/interface/RPCAMCLinkEvent.h"

class RPCAMCLinkEvents {
public:
  // from FED CDF Header and Trailer
  static unsigned int const fed_event_ = RPCAMCLinkEvent::fed_ | RPCAMCLinkEvent::debug_ | 0;
  static unsigned int const fed_header_check_fail_ = RPCAMCLinkEvent::fed_ | RPCAMCLinkEvent::warn_ | 1;
  static unsigned int const fed_header_id_mismatch_ = RPCAMCLinkEvent::fed_ | RPCAMCLinkEvent::warn_ | 2;
  static unsigned int const fed_trailer_check_fail_ = RPCAMCLinkEvent::fed_ | RPCAMCLinkEvent::warn_ | 3;
  static unsigned int const fed_trailer_length_mismatch_ = RPCAMCLinkEvent::fed_ | RPCAMCLinkEvent::warn_ | 4;
  static unsigned int const fed_trailer_crc_mismatch_ = RPCAMCLinkEvent::fed_ | RPCAMCLinkEvent::warn_ | 5;
  // from AMC13 Header
  static unsigned int const fed_amc13_block_incomplete_ = RPCAMCLinkEvent::fed_ | RPCAMCLinkEvent::warn_ | 6;
  // from AMC13 AMC Header
  static unsigned int const fed_amc13_amc_number_invalid_ = RPCAMCLinkEvent::fed_ | RPCAMCLinkEvent::warn_ | 7;
  static unsigned int const amc_amc13_block_incomplete_ = RPCAMCLinkEvent::amc_ | RPCAMCLinkEvent::warn_ | 1;

  static unsigned int const amc_event_ = RPCAMCLinkEvent::amc_ | RPCAMCLinkEvent::debug_ | 0;
  static unsigned int const amc_amc13_evc_bc_invalid_ = RPCAMCLinkEvent::amc_ | RPCAMCLinkEvent::warn_ | 2;
  static unsigned int const amc_amc13_length_incorrect_ = RPCAMCLinkEvent::amc_ | RPCAMCLinkEvent::warn_ | 3;
  static unsigned int const amc_amc13_crc_mismatch_ = RPCAMCLinkEvent::amc_ | RPCAMCLinkEvent::warn_ | 4;
  static unsigned int const amc_amc13_size_inconsistent_ = RPCAMCLinkEvent::amc_ | RPCAMCLinkEvent::warn_ | 5;
  static unsigned int const amc_payload_incomplete_ = RPCAMCLinkEvent::amc_ | RPCAMCLinkEvent::warn_ | 6;
  // from AMC Payload Header
  static unsigned int const amc_number_mismatch_ = RPCAMCLinkEvent::amc_ | RPCAMCLinkEvent::warn_ | 7;
  static unsigned int const amc_size_mismatch_ = RPCAMCLinkEvent::amc_ | RPCAMCLinkEvent::warn_ | 8;

  // from RPC Record
  static unsigned int const amc_link_invalid_ = RPCAMCLinkEvent::amc_ | RPCAMCLinkEvent::warn_ | 9;

  static unsigned int const input_event_ = RPCAMCLinkEvent::input_ | RPCAMCLinkEvent::debug_ | 0;
  static unsigned int const input_link_error_ = RPCAMCLinkEvent::input_ | RPCAMCLinkEvent::warn_ | 1;
  static unsigned int const input_link_ack_fail_ = RPCAMCLinkEvent::input_ | RPCAMCLinkEvent::warn_ | 2;
  static unsigned int const input_eod_ = RPCAMCLinkEvent::input_ | RPCAMCLinkEvent::info_ | 3;
  static unsigned int const input_lb_invalid_ = RPCAMCLinkEvent::input_ | RPCAMCLinkEvent::warn_ | 4;
  static unsigned int const input_connector_invalid_ = RPCAMCLinkEvent::input_ | RPCAMCLinkEvent::warn_ | 5;
  static unsigned int const input_connector_not_used_ = RPCAMCLinkEvent::input_ | RPCAMCLinkEvent::warn_ | 6;

  static unsigned int const fed_min_ = 0;
  static unsigned int const fed_max_ = 8;
  static unsigned int const amc_min_ = 0;
  static unsigned int const amc_max_ = 10;
  static unsigned int const input_min_ = 0;
  static unsigned int const input_max_ = 7;

public:
  RPCAMCLinkEvents();

  static std::string getEventName(unsigned int event);
};

#endif  // EventFilter_RPCRawToDigi_RPCAMCLinkEvents_h
