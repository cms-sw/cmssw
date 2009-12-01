#ifndef CondCore_TagCollection_PfnEditor
#define CondCore_TagCollection_PfnEditor


#include <string>

namespace cond{

  // edit the pfn accordind to rules given in its constructor
  class PfnEditor {
  public:
    PfnEditor(std::string const & ipre, 
	      std::string const & ipos) : 
      prefix(ipre), 
      postfix(ipos),
      off(prefix.empty() && postfix.empty())
    {}
    
    
    std::string operator(std::string const & pfn) {
      if (off) return pfn;
      size_t pos=std::string::npos;
      if (!prefix.empty()) pos = pfn.rfind('/');
      return ( (pos == std::string::npos) ? (prefix+pfn) :
	       pfn.replace(0,pos,prefix)
	       ) + postfix;
    }

  private:
    std::string prefix;
    std::string postfix;
    bool off;
  };


}
#endif // CondCore_TagCollection_PfnEditor

