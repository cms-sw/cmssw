#include <string>

#include "HLTrigger/HLTcore/interface/TriggerExpressionParser.h"

int parseExpression(std::string const& expression) {
  std::cout << "Input expression: \"" << expression << "\"\n";

  auto const* eval = triggerExpression::parse(expression);

  auto ret = 0;
  if (not eval) {
    std::cout << "Parsing failed.\n";
    ++ret;
  } else {
    std::cout << "Parsing output:   \"" << *eval << "\"\n";
  }

  std::cout << "--------------------------------------------------------\n";

  return ret;
}

void showHelpMessage() {
  std::cout << "\n"
            << "Purpose:\n"
            << "  test parsing of N expressions with the triggerExpression::Parser\n\n"
            << "Input:\n"
            << "  N command-line arguments, each parsed as a separate expression\n"
            << "  (if no arguments are given, or \"-h\" or \"--help\" is specified,"
            << " this help message is shown)\n\n"
            << "Exit code:\n"
            << "  number of expressions for which parsing failed\n\n"
            << "Example:\n"
            << "  > hltParseTriggerExpressions \"EXPR1\" \"EXPR2\" [..]\n\n";
}

int main(int argc, char** argv) {
  std::cout << "===             hltParseTriggerExpressions           ===\n";
  std::cout << "=== parse expressions with triggerExpression::Parser ===\n";
  std::cout << "--------------------------------------------------------\n";

  if (argc < 2) {
    std::cout << "No expressions specified for parsing. See help message below.\n";
    showHelpMessage();
    return 0;
  }

  std::vector<std::string> v_exprs;
  v_exprs.reserve(argc - 1);
  for (auto idx = 1; idx < argc; ++idx) {
    v_exprs.emplace_back(argv[idx]);
    if (v_exprs.back() == "-h" or v_exprs.back() == "--help") {
      showHelpMessage();
      v_exprs.clear();
      break;
    }
  }

  auto ret = 0;
  for (auto const& expr : v_exprs) {
    ret += parseExpression(expr);
  }

  return ret;
}
