#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabaseImpl.hh"

#ifndef HAVE_XDAQ
#include "CaloOnlineTools/HcalOnlineDb/interface/xdaq_compat.h"  // Includes typedef for log4cplus::Logger
#endif

#include <iostream>  // std::cout

namespace hcal {

  ConfigurationDatabaseImpl::ConfigurationDatabaseImpl() : m_logger(&std::cout) {}

  void ConfigurationDatabaseImpl::parseAccessor(const std::string& accessor,
                                                std::string& method,
                                                std::string& host,
                                                std::string& port,
                                                std::string& user,
                                                std::string& db,
                                                std::map<std::string, std::string>& params) {
    std::string::size_type start, end;

    method.clear();
    host.clear();
    port.clear();
    user.clear();
    db.clear();
    params.clear();

    if (accessor.empty())
      return;

    // method
    start = 0;
    end = accessor.find("://");
    if (end == std::string::npos)
      return;

    method = accessor.substr(start, end - start);
    start = end + 3;  // skip past ://

    end = accessor.find('@', start);  // user?
    if (end != std::string::npos) {
      user = accessor.substr(start, end - start);
      start = end + 1;
    }

    end = accessor.find(':', start);  // port?
    if (end != std::string::npos) {
      host = accessor.substr(start, end - start);  // host is here, port is next
      start = end + 1;
    }

    end = accessor.find('/', start);  // port or host
    if (end == std::string::npos)
      return;  // problems...
    if (host.empty())
      host = accessor.substr(start, end - start);
    else
      port = accessor.substr(start, end - start);
    start = end + 1;

    end = accessor.find('?', start);  // database
    if (end == std::string::npos) {
      db = accessor.substr(start);
      return;
    } else
      db = accessor.substr(start, end - start);
    start = end;  //  beginning of the parameters

    // parameters
    std::string pname, pvalue;
    while (start != std::string::npos) {
      start += 1;
      end = accessor.find('=', start);
      if (end == std::string::npos)
        break;  // no equals
      pname = accessor.substr(start, end - start);
      start = end + 1;
      end = accessor.find_first_of(",&", start);
      if (end == std::string::npos) {
        pvalue = accessor.substr(start);
      } else {
        pvalue = accessor.substr(start, end - start);
      }
      params[pname] = pvalue;
      start = end;
    }
  }

  std::vector<std::string> ConfigurationDatabaseImpl::getValidTags() noexcept(false) {
    XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException, "Not implemented");
  }
  ConfigurationDatabase::ApplicationConfig ConfigurationDatabaseImpl::getApplicationConfig(
      const std::string& tag, const std::string& classname, int instance) noexcept(false) {
    XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException, "Not implemented");
  }

  std::string ConfigurationDatabaseImpl::getConfigurationDocument(const std::string& tag) noexcept(false) {
    XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException, "Not implemented");
  }

  unsigned int ConfigurationDatabaseImpl::getFirmwareChecksum(const std::string& board,
                                                              unsigned int version) noexcept(false) {
    XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException, "Not implemented");
  }

  void ConfigurationDatabaseImpl::getFirmwareMCS(const std::string& board,
                                                 unsigned int version,
                                                 std::vector<std::string>& mcsLines) noexcept(false) {
    XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException, "Not implemented");
  }
  void ConfigurationDatabaseImpl::getLUTs(
      const std::string& tag,
      int crate,
      int slot,
      std::map<ConfigurationDatabase::LUTId, ConfigurationDatabase::LUT>& LUTs) noexcept(false) {
    XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException, "Not implemented");
  }
  void ConfigurationDatabaseImpl::getLUTChecksums(
      const std::string& tag,
      std::map<ConfigurationDatabase::LUTId, ConfigurationDatabase::MD5Fingerprint>& checksums) noexcept(false) {
    XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException, "Not implemented");
  }
  void ConfigurationDatabaseImpl::getPatterns(
      const std::string& tag,
      int crate,
      int slot,
      std::map<ConfigurationDatabase::PatternId, ConfigurationDatabase::HTRPattern>& patterns) noexcept(false) {
    XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException, "Not implemented");
  }
  void ConfigurationDatabaseImpl::getZSThresholds(
      const std::string& tag,
      int crate,
      int slot,
      std::map<ConfigurationDatabase::ZSChannelId, int>& thresholds) noexcept(false) {
    XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException, "Not implemented");
  }
  void ConfigurationDatabaseImpl::getHLXMasks(
      const std::string& tag,
      int crate,
      int slot,
      std::map<ConfigurationDatabase::FPGAId, ConfigurationDatabase::HLXMasks>& masks) noexcept(false) {
    XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException, "Not implemented");
  }
  void ConfigurationDatabaseImpl::getRBXdata(
      const std::string& tag,
      const std::string& rbx,
      ConfigurationDatabase::RBXdatumType dtype,
      std::map<ConfigurationDatabase::RBXdatumId, ConfigurationDatabase::RBXdatum>& RBXdata) noexcept(false) {
    XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException, "Not implemented");
  }
  void ConfigurationDatabaseImpl::getRBXpatterns(
      const std::string& tag,
      const std::string& rbx,
      std::map<ConfigurationDatabase::RBXdatumId, ConfigurationDatabase::RBXpattern>& patterns) noexcept(false) {
    XCEPT_RAISE(hcal::exception::ConfigurationDatabaseException, "Not implemented");
  }

  // added by Gena Kukartsev
  oracle::occi::Connection* ConfigurationDatabaseImpl::getConnection(void) { return nullptr; }

  oracle::occi::Environment* ConfigurationDatabaseImpl::getEnvironment(void) { return nullptr; }

}  // namespace hcal
