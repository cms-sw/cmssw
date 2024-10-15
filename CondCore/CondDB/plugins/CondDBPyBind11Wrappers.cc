#include "CondCore/CondDB/interface/CredentialStore.h"
#include "CondCore/CondDB/interface/Auth.h"
//
#include <cstdlib>
#include <cassert>

namespace cond {

  std::tuple<std::string, std::string, std::string> getDbCredentials(const std::string& connectionString,
                                                                     int accessType,
                                                                     const std::string& authPath) {
    std::string ap = authPath;
    if (ap.empty()) {
      char* auth_path_env = std::getenv(cond::auth::COND_AUTH_PATH);
      assert(auth_path_env);
      ap = std::string(auth_path_env);
    }
    auto ret = std::make_tuple(std::string(""), std::string(""), std::string(""));
    if (!ap.empty()) {
      CredentialStore credDb;
      credDb.setUpForConnectionString(connectionString, ap);
      std::string role(cond::auth::s_roleCodeArray[accessType].first);
      auto creds = credDb.getUserCredentials(connectionString, role);
      ret = std::tie(credDb.keyPrincipalName(), creds.first, creds.second);
    }
    return ret;
  }

}  // namespace cond

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(libCondDBPyBind11Interface, m) {
  m.def("get_credentials_from_db", &cond::getDbCredentials, "Get db credentials for a connection string");
  m.attr("default_role") = pybind11::int_(int(cond::auth::DEFAULT_ROLE));
  m.attr("reader_role") = pybind11::int_(int(cond::auth::READER_ROLE));
  m.attr("writer_role") = pybind11::int_(int(cond::auth::WRITER_ROLE));
  m.attr("admin_role") = pybind11::int_(int(cond::auth::ADMIN_ROLE));
}
