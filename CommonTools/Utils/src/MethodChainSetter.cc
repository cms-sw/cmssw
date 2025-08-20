#include "CommonTools/Utils/interface/parser/MethodChainSetter.h"
#include "CommonTools/Utils/interface/parser/MethodChain.h"
#include "CommonTools/Utils/interface/returnType.h"
#include "CommonTools/Utils/interface/parser/Exception.h"
#include <string>

using namespace reco::parser;
using namespace std;

void MethodChainSetter::operator()(const char *begin, const char *end) const {
  //std::cerr << "MethodChainSetter: Pushed [" << std::string(begin,end) << "]" << std::endl;
  if (!methStack_.empty())
    push(begin, end);
  else if (!lazyMethStack_.empty())
    lazyPush(begin, end);
  else
    throw Exception(begin) << " Expression didn't parse neither hastily nor lazyly. This must not happen.\n";
}

void MethodChainSetter::push(const char *begin, const char *end) const {
  methchain_ = std::shared_ptr<MethodChainBase>(new MethodChain(methStack_));
  methStack_.clear();
  typeStack_.resize(1);
}

void MethodChainSetter::lazyPush(const char *begin, const char *end) const {
  methchain_ = std::shared_ptr<MethodChainBase>(new LazyMethodChain(lazyMethStack_));
  lazyMethStack_.clear();
  typeStack_.resize(1);
}
