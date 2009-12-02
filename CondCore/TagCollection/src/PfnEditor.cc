#include "CondCore/TagCollection/interface/PfnEditor.h"

namespace cond{
  
  PfnEditor::PfnEditor() : off(true){}
  
PfnEditor:PfnEditor(std::string const & ipre, 
		    std::string const & ipos) : 
  prefix(ipre), 
  postfix(ipos),
  off(prefix.empty() && postfix.empty())
{}
  
  
  std::string PfnEditor:operator()(std::string const & pfn) {
    if (off) return pfn;
    // FIXME ad-hoc
    if (pfn.find("FrontierInt")!=std::string::npos)  return pfn;
 
   size_t pos=std::string::npos;
    if (!prefix.empty()) pos = pfn.rfind('/');
    return prefix + ( (pos == std::string::npos) ? pfn :
		      pfn.substr(pos+1)
		      ) + postfix;
  }
  
}
