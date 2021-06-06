#include "CondCore/CondDB/interface/Exception.h"
#include "CondCore/CondDB/interface/Utils.h"
#include "DbConnectionString.h"
//
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

namespace cond {

  namespace persistency {

    unsigned int countslash(const std::string& input) {
      unsigned int count = 0;
      std::string::size_type slashpos(0);
      while (slashpos != std::string::npos) {
        slashpos = input.find('/', slashpos);
        if (slashpos != std::string::npos) {
          ++count;
          // start next search after this word
          slashpos += 1;
        }
      }
      return count;
    }

    std::string parseFipConnectionString(const std::string& fipConnect) {
      std::string connect("sqlite_file:");
      std::string::size_type pos = fipConnect.find(':');
      std::string fipLocation = fipConnect.substr(pos + 1);
      edm::FileInPath fip(fipLocation);
      connect.append(fip.fullPath());
      return connect;
    }

    //FIXME: sdg this function does not support frontier connections strings like
    //frontier://cmsfrontier.cern.ch:8000/FrontierPrep/CMS_CONDITIONS
    //as http://cmsfrontier.cern.ch:8000/FrontierPrep(freshkey=foo) is an invalid URI.
    std::pair<std::string, std::string> getConnectionParams(const std::string& connectionString,
                                                            const std::string& transactionId,
                                                            const std::string& signature) {
      if (connectionString.empty())
        cond::throwException("The connection string is empty.", "getConnectionParams");
      std::string protocol = getConnectionProtocol(connectionString);
      std::string finalConn = connectionString;
      std::string refreshConn("");
      if (protocol == "frontier") {
        std::string protocol("frontier://");
        std::string::size_type fpos = connectionString.find(protocol);
        unsigned int nslash = countslash(connectionString.substr(protocol.size(), connectionString.size() - fpos));
        if (nslash == 1) {
          edm::Service<edm::SiteLocalConfig> localconfservice;
          if (!localconfservice.isAvailable()) {
            cond::throwException("edm::SiteLocalConfigService is not available", "getConnectionParams");
          }
          finalConn = localconfservice->lookupCalibConnect(connectionString);
        }
        if (!transactionId.empty()) {
          size_t l = finalConn.rfind('/');
          finalConn.insert(l, "(freshkey=" + transactionId + ')');
        }

        //When the signature parameter is set to sig, FroNTier requests that the server sends digital signatures on every response.
        //We test here that the signature string, if defined, is actually set to sig, otherwise we throw an exception
        std::string signatureParameter("sig");
        if (!signature.empty()) {
          if (signature == signatureParameter) {
            std::string::size_type s = finalConn.rfind('/');
            finalConn.insert(s, "(security=" + signature + ')');
          } else {
            cond::throwException("The FroNTier security option is invalid.", "getConnectionParams");
          }
        }

        std::string::size_type startRefresh = finalConn.find("://");
        if (startRefresh != std::string::npos) {
          startRefresh += 3;
        }
        std::string::size_type endRefresh = finalConn.rfind('/', std::string::npos);
        if (endRefresh == std::string::npos) {
          refreshConn = finalConn;
        } else {
          refreshConn = finalConn.substr(startRefresh, endRefresh - startRefresh);
          if (refreshConn.substr(0, 1) != "(") {
            //if the connect string is not a complicated parenthesized string,
            // an http:// needs to be at the beginning of it
            refreshConn.insert(0, "http://");
          }
        }
      } else if (protocol == "sqlite_fip") {
        finalConn = parseFipConnectionString(connectionString);
      }
      return std::make_pair(finalConn, refreshConn);
    }

  }  // namespace persistency
}  // namespace cond
