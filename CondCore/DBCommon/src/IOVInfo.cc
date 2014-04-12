#include "CondCore/DBCommon/interface/IOVInfo.h"

#include <unistd.h>
#include <cstdlib>
#include <sstream>
#include <vector>
namespace cond {

  std::string userInfo() {
    // this are really static stuff
    std::ostringstream user_info;
    char * user= ::getenv("USER");
    std::vector<char> hname(1024,'\0');
    char * hostname = &hname.front();
    ::gethostname(hostname, 1024);
    char * pwd = ::getenv("PWD");
    if (user) { user_info<< "USER=" << user <<";" ;} else { user_info<< "USER="<< "??;";}
    if (hostname[0] != '\0') {user_info<< "HOSTNAME=" << hostname <<";";} 
    else { user_info<< "HOSTNAME="<< "??;";}
    if (pwd) {user_info<< "PWD=" << pwd <<";";} else  {user_info<< "PWD="<< "??;";}
    return user_info.str();
  }
  
}
