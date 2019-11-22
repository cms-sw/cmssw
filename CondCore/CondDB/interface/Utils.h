#ifndef CondCore_CondDB_Utils_h
#define CondCore_CondDB_Utils_h

#include "CondCore/CondDB/interface/Exception.h"
//
#include <string>
#include <cxxabi.h>
#include <algorithm>
#include <iostream>
#include <tuple>
#include <fstream>
#include <unistd.h>
#include <pwd.h>
#include <climits>
//
#include <boost/regex.hpp>

namespace cond {

  namespace {

    inline std::string demangledName(const std::type_info& typeInfo) {
      int status = 0;
      std::string ret("");
      char* realname = abi::__cxa_demangle(typeInfo.name(), nullptr, nullptr, &status);
      if (status == 0 && realname) {
        ret = realname;
        free(realname);
      }
      // clean up the spaces... ( to be removed after getting rid of reflex - that does not have spaces...)
      ret.erase(std::remove(ret.begin(), ret.end(), ' '), ret.end());
      return ret;
    }

    inline std::string currentCMSSWVersion() {
      std::string version("");
      const char* envVersion = std::getenv("CMSSW_VERSION");
      if (envVersion) {
        version += envVersion;
      }
      return version;
    }

    inline std::string currentArchitecture() {
      std::string arch("");
      const char* archEnv = std::getenv("SCRAM_ARCH");
      if (archEnv) {
        arch += archEnv;
      }
      return arch;
    }

    inline std::string getUserName() {
      struct passwd* user_creds = getpwuid(getuid());
      if (user_creds == nullptr)
        return std::string("USER_NOT_AVAILABLE");
      return std::string(user_creds->pw_name);
    }

    inline std::string getHostName() {
      char hostname[HOST_NAME_MAX];
      int retcode = gethostname(hostname, HOST_NAME_MAX);
      if (retcode)
        return "";
      return std::string(hostname);
    }

    inline std::string getCommand() {
      std::string commName("");
      try {
        std::ifstream comm("/proc/self/cmdline");
        std::getline(comm, commName);
        size_t ind = commName.find('\0');
        while (ind != std::string::npos) {
          commName.replace(ind, 1, 1, ' ');
          ind = commName.find('\0');
        }
      } catch (std::ifstream::failure const&) {
        commName = "unknown";
      }
      return commName;
    }
  }  // namespace

  namespace persistency {

    inline std::string getConnectionProtocol(const std::string& connectionString) {
      size_t techEnd = connectionString.find(':');
      if (techEnd == std::string::npos)
        throwException("Could not resolve the connection protocol on " + connectionString + ".",
                       "getConnectionProtocol");
      std::string technology = connectionString.substr(0, techEnd);
      return technology;
    }

    inline std::tuple<std::string, std::string, std::string> parseConnectionString(const std::string& connectionString) {
      std::string protocol = getConnectionProtocol(connectionString);
      std::string serviceName("");
      std::string databaseName("");
      if (protocol == "sqlite" || protocol == "sqlite_file" || protocol == "sqlite_fip") {
        databaseName = connectionString.substr(protocol.size() + 1);
      } else if (protocol == "oracle" || protocol == "frontier") {
        size_t ptr = protocol.size() + 1;
        if (connectionString.substr(ptr, 2) != "//")
          throwException(
              "Connection string " + connectionString + " is invalid format for technology \"" + protocol + "\".",
              "parseConnectionString");
        ptr += 2;
        size_t serviceEnd = connectionString.find('/', ptr);
        if (serviceEnd == std::string::npos)
          throwException("Connection string " + connectionString + " is invalid.", "parseConnectionString");
        serviceName = connectionString.substr(ptr, serviceEnd - ptr);
        ptr = serviceEnd + 1;
        databaseName = connectionString.substr(ptr);
      } else
        throwException("Technology " + protocol + " is not known.", "parseConnectionString");

      return std::make_tuple(protocol, serviceName, databaseName);
    }

    inline std::string convertoToOracleConnection(const std::string& input) {
      // leave the connection string unmodified for sqlite
      if (input.find("sqlite") == 0 || input.find("oracle") == 0)
        return input;

      //static const boost::regex trivial("oracle://(cms_orcon_adg|cms_orcoff_prep)/([_[:alnum:]]+?)");
      static const boost::regex short_frontier("frontier://([[:alnum:]]+?)/([_[:alnum:]]+?)");
      static const boost::regex long_frontier("frontier://((\\([-[:alnum:]]+?=[^\\)]+?\\))+)/([_[:alnum:]]+?)");
      static const boost::regex long_frontier_serverurl("\\(serverurl=[^\\)]+?/([[:alnum:]]+?)\\)");

      static const std::map<std::string, std::string> frontierMap = {
          {"PromptProd", "cms_orcon_adg"},
          {"FrontierProd", "cms_orcon_adg"},
          {"FrontierArc", "cms_orcon_adg"},
          {"FrontierOnProd", "cms_orcon_adg"},
          {"FrontierPrep", "cms_orcoff_prep"},
      };

      boost::smatch matches;

      static const std::string technology("oracle://");
      std::string service("");
      std::string account("");

      bool match = false;
      if (boost::regex_match(input, matches, short_frontier)) {
        service = matches[1];
        account = matches[2];
        match = true;
      }

      if (boost::regex_match(input, matches, long_frontier)) {
        std::string frontier_config(matches[1]);
        boost::smatch matches2;
        if (not boost::regex_search(frontier_config, matches2, long_frontier_serverurl))
          throwException("No serverurl in matched long frontier", "convertoToOracleConnection");
        service = matches2[1];
        account = matches[3];
        match = true;
      }

      if (!match)
        throwException("Connection string " + input + " can't be converted to oracle connection.",
                       "convertoToOracleConnection");

      if (service == "FrontierArc") {
        size_t len = account.size() - 5;
        account = account.substr(0, len);
      }

      auto it = frontierMap.find(service);
      if (it == frontierMap.end())
        throwException("Connection string can't be converted.", "convertoToOracleConnection");
      service = it->second;

      return technology + service + "/" + account;
    }

  }  // namespace persistency

}  // namespace cond

#endif
