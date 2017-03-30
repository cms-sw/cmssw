#ifndef CondTools_RPC_RPCTwinMuxLinkMapHandler_h
#define CondTools_RPC_RPCTwinMuxLinkMapHandler_h

#include <string>
#include <vector>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/Common/interface/Time.h"

#include "CondFormats/RPCObjects/interface/RPCAMCLinkMap.h"

namespace edm {
class ParameterSet;
class ConfigurationDescriptions;
} // namespace edm

class RPCTwinMuxLinkMapHandler
    : public popcon::PopConSourceHandler<RPCAMCLinkMap>
{
public:
    RPCTwinMuxLinkMapHandler(edm::ParameterSet const & config);
    ~RPCTwinMuxLinkMapHandler();

    void getNewObjects();
    std::string id() const;

protected:
    std::string id_;
    std::string data_tag_;
    cond::Time_t since_run_;

    std::string input_file_;
    std::vector<int> wheel_fed_;
    std::vector<std::vector<int> > wheel_sector_amc_;

    std::string txt_file_;
};

#endif // CondTools_RPC_RPCTwinMuxLinkMapHandler_h
