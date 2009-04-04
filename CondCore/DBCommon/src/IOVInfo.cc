#include "CondCore/DBCommon/interface/IOVInfo.h"

#include <cstdlib>
#include <sstream>

namespace cond {

  std::string userInfo() {
    // this are really static stuff
    std::ostringstream user_info;
    char * user= ::getenv("USER");
    char * hostname= ::getenv("HOSTNAME");
    char * pwd = ::getenv("PWD");
    if (user) { user_info<< "USER=" << user <<";" ;} else { user_info<< "USER="<< "??;";}
    if (hostname) {user_info<< "HOSTNAME=" << hostname <<";";} else { user_info<< "HOSTNAME="<< "??;";}
    if (pwd) {user_info<< "PWD=" << pwd <<";";} else  {user_info<< "PWD="<< "??;";}
    return user_info.str();
  }
  
}
