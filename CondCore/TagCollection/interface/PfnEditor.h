#ifndef CondCore_TagCollection_PfnEditor
#define CondCore_TagCollection_PfnEditor


#include <string>

namespace cond{

  // edit the pfn accordind to rules given in its constructor
  class PfnEditor {
  public:
    PfnEditor();
    PfnEditor(std::string const & ipre, 
	      std::string const & ipos);
    
    std::string operator()(std::string const & pfn) const;

  private:
    std::string prefix;
    std::string postfix;
    bool off;
  };


}
#endif // CondCore_TagCollection_PfnEditor

