#include "CondCore/CondDB/interface/CredentialStore.h"
#include "CondCore/CondDB/interface/Auth.h"
//
#include <cstdlib>

namespace cond {

  std::pair<std::string, std::string> getDbCredentials(const std::string& connectionString, bool updateMode, const std::string& authPath){
    std::string ap = authPath;
    if(ap==""){
      ap = std::string(std::getenv(cond::auth::COND_AUTH_PATH));
    }
    auto ret = std::make_pair(std::string(""),std::string(""));
    if(!ap.empty()){ 
      CredentialStore credDb;
      credDb.setUpForConnectionString( connectionString, ap );
      std::string role(cond::auth::COND_READER_ROLE);
      if( updateMode ) role = cond::auth::COND_WRITER_ROLE;
      ret = credDb.getUserCredentials( connectionString, role );
    }
    return ret;
  }

}  // namespace cond

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(pluginCondDBPyBind11Interface, m) {
  m.def("get_db_credentials", &cond::getDbCredentials, "Get db credentials for a connection string");
}

