#ifndef CommonTools_Utils_FunctionSetter_h
#define CommonTools_Utils_FunctionSetter_h
/* \class reco::parser::FunctionSetter
 *
 * Function setter
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */
#include "CommonTools/Utils/interface/parser/Function.h"
#include "CommonTools/Utils/interface/parser/FunctionStack.h"

namespace reco {
  namespace parser {
    struct FunctionSetter {
      FunctionSetter(Function fun, FunctionStack& stack) : fun_(fun), stack_(stack) {}

      void operator()(const char*, const char*) const {
#ifdef BOOST_SPIRIT_DEBUG
        BOOST_SPIRIT_DEBUG_OUT << "pushing math function: " << functionNames[fun_] << std::endl;
#endif
        stack_.push_back(fun_);
      }

    private:
      Function fun_;
      FunctionStack& stack_;
    };

    struct FunctionSetterCommit {
      FunctionSetterCommit(FunctionStack& stackFrom, FunctionStack& stackTo) : from_(stackFrom), to_(stackTo) {}
      void operator()(const char&) const {
        to_.push_back(from_.back());
        from_.clear();
      }

    private:
      FunctionStack& from_;
      FunctionStack& to_;
    };
  }  // namespace parser
}  // namespace reco

#endif
