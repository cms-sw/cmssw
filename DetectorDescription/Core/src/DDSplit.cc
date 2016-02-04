
#include "DetectorDescription/Core/interface/DDSplit.h"

std::pair<std::string,std::string> DDSplit(const std::string & n) 
{
  std::string name,ns;
  std::string::size_type pos = n.find(':');
  if (pos==std::string::npos) {
    ns = "";
    name = n;
  }
  else {
    ns = std::string(n,0,pos);
    name = std::string(n,pos+1,n.size()-1);
  }    
  return std::make_pair(name,ns);
}

