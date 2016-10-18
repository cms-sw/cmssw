#ifndef CondTools_RPC_RPCDCCLinkMapHandler_h
#define CondTools_RPC_RPCDCCLinkMapHandler_h

#include <string>

#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/Common/interface/Time.h"

#include "CondFormats/RPCObjects/interface/RPCDCCLinkMap.h"

namespace edm {
class ParameterSet;
class ConfigurationDescriptions;
} // namespace edm

class RPCDCCLinkMapHandler
    : public popcon::PopConSourceHandler<RPCDCCLinkMap>
{
public:
    RPCDCCLinkMapHandler(edm::ParameterSet const & _config);
    ~RPCDCCLinkMapHandler();

    void getNewObjects();
    std::string id() const;

protected:
    std::string id_;
    std::string data_tag_;
    cond::Time_t since_run_;

    std::string txt_file_;

    std::string connect_;
    cond::persistency::ConnectionPool connection_;
};

#endif // CondTools_RPC_RPCDCCLinkMapHandler_h
