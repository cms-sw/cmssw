//local includes
#include "CondCore/DBCommon/interface/Auth.h"

namespace cond{

  const char* Auth::COND_AUTH_PATH = "COND_AUTH_PATH";
  const char* Auth::COND_AUTH_SYS = "COND_AUTH_SYS";

  const std::string Auth::COND_ADMIN_GROUP("COND_ADMIN_GROUP");

  const std::string Auth::COND_DEFAULT_ROLE("COND_DEFAULT_ROLE");
  const std::string Auth::COND_WRITER_ROLE("COND_WRITER_ROLE");
  const std::string Auth::COND_READER_ROLE("COND_READER_ROLE");
  const std::string Auth::COND_ADMIN_ROLE("COND_ADMIN_ROLE");
  const std::string Auth::COND_DEFAULT_PRINCIPAL("COND_DEFAULT_PRINCIPAL");
  const std::string Auth::COND_KEY("Memento");

  // horrible workaround: coral does not allow to define custom properties: only the pre-defined ones can be used! In this case, AuthenticationFile
  const std::string Auth::COND_AUTH_PATH_PROPERTY("AuthenticationFile");


}
