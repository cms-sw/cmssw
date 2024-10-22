#include "CommonTools/Utils/interface/parser/Grammar.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "CommonTools/Utils/interface/expressionParser.h"

using namespace reco::parser;

bool reco::parser::expressionParser(const edm::TypeWithDict& t,
                                    const std::string& value,
                                    ExpressionPtr& expr,
                                    bool lazy) {
  using namespace boost::spirit::classic;
  Grammar grammar(expr, t, lazy);
  bool returnValue = false;
  const char* startingFrom = value.c_str();
  try {
    returnValue = parse(startingFrom, grammar.use_parser<1>() >> end_p, space_p).full;
  } catch (BaseException& e) {
    throw edm::Exception(edm::errors::Configuration)
        << "Expression parser error:" << baseExceptionWhat(e) << " (char " << e.where - startingFrom << ")\n";
  }
  return returnValue;
}
