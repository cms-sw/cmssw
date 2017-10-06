#ifndef CondTools_RPC_RPCAMCLinkMapHandler_h
#define CondTools_RPC_RPCAMCLinkMapHandler_h

#include <string>
#include <vector>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/Common/interface/Time.h"

#include "CondFormats/RPCObjects/interface/RPCAMCLinkMap.h"

namespace edm {
class ParameterSet;
class ConfigurationDescriptions;
} // namespace edm

class RPCAMCLinkMapHandler
    : public popcon::PopConSourceHandler<RPCAMCLinkMap>
{
public:
    RPCAMCLinkMapHandler(edm::ParameterSet const & config);
    ~RPCAMCLinkMapHandler() override;

    void getNewObjects() override;
    std::string id() const override;

protected:
    std::string id_;
    std::string data_tag_;
    cond::Time_t since_run_;

    std::string input_file_;
    bool wheel_not_side_;
    std::vector<int> wos_fed_;
    unsigned int n_sectors_;
    std::vector<std::vector<int> > wos_sector_amc_;

    std::string txt_file_;
};

#endif // CondTools_RPC_RPCAMCLinkMapHandler_h
