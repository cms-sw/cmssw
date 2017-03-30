#ifndef CondTools_RPC_RPCCPPFLinkMapHandler_h
#define CondTools_RPC_RPCCPPFLinkMapHandler_h

#include <string>
#include <vector>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/Common/interface/Time.h"

#include "CondFormats/RPCObjects/interface/RPCAMCLinkMap.h"

namespace edm {
class ParameterSet;
class ConfigurationDescriptions;
} // namespace edm

class RPCCPPFLinkMapHandler
    : public popcon::PopConSourceHandler<RPCAMCLinkMap>
{
public:
    RPCCPPFLinkMapHandler(edm::ParameterSet const & config);
    ~RPCCPPFLinkMapHandler();

    void getNewObjects();
    std::string id() const;

protected:
    std::string id_;
    std::string data_tag_;
    cond::Time_t since_run_;

    std::string input_file_;
    std::vector<int> side_fed_;
    unsigned int n_sectors_;
    std::vector<std::vector<int> > side_sector_amc_;

    std::string txt_file_;
};

#endif // CondTools_RPC_RPCCPPFLinkMapHandler_h
