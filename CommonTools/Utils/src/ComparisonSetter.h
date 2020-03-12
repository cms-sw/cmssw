#ifndef CommonTools_Utils_ComparisonSetter_h
#define CommonTools_Utils_ComparisonSetter_h
/* \class reco::parser::ComparisonSetter
 *
 * Comparison setter
 *
 * \author original version: Chris Jones, Cornell, 
 *         adapted by Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */
#include "CommonTools/Utils/src/ComparisonStack.h"
#include "CommonTools/Utils/src/Comparison.h"
#ifdef BOOST_SPIRIT_DEBUG
#include <iostream>
#include <string>
#endif
#include <memory>

namespace reco {
  namespace parser {
#ifdef BOOST_SPIRIT_DEBUG
    template <typename Op>
    struct cmp_out {
      static const std::string value;
    };
#endif

    template <class CompT>
    struct ComparisonSetter {
      ComparisonSetter(ComparisonStack& stack) : stack_(stack) {}
      void operator()(const char) const {
#ifdef BOOST_SPIRIT_DEBUG
        BOOST_SPIRIT_DEBUG_OUT << "pushing comparison: " << cmp_out<CompT>::value << std::endl;
#endif
        stack_.push_back(std::shared_ptr<ComparisonBase>(new Comparison<CompT>()));
      }

    private:
      ComparisonStack& stack_;
    };
  }  // namespace parser
}  // namespace reco

#endif
