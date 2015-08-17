#ifndef Parser_SelectorPtr_h
#define Parser_SelectorPtr_h
/* \class reco::parser::SelectorPtr
 *
 * Shared pointer to selector
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */
#include <memory>

namespace reco {  
  namespace parser {
    class SelectorBase;
    typedef std::shared_ptr<const SelectorBase> SelectorPtr;
  }  
  template<typename T> class CutOnObject;
  namespace exprEval {
  template<typename T>
  struct do_not_delete{
    void operator()(T* p) const { }
  };
  template<typename T> 
    using SelectorPtr = std::unique_ptr<const reco::CutOnObject<T>, do_not_delete<const reco::CutOnObject<T> > >;
  };
}

#endif
