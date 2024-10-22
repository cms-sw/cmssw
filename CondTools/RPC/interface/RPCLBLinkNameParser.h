#ifndef CondTools_RPC_RPCLBLinkNameParser_h
#define CondTools_RPC_RPCLBLinkNameParser_h

#include <string>

#include "CondFormats/RPCObjects/interface/RPCLBLink.h"

class RPCLBLinkNameParser {
public:
  static void parse(std::string const& name, RPCLBLink& lb_link);
  static RPCLBLink parse(std::string const& name);
};

#endif  // CondTools_RPC_RPCLBLinkNameParser_h
