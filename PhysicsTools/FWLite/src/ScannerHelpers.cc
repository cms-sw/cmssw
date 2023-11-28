#include <PhysicsTools/FWLite/interface/ScannerHelpers.h>

#include "CommonTools/Utils/interface/parser/ExpressionPtr.h"
#include "CommonTools/Utils/interface/parser/Grammar.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <iostream>

reco::parser::ExpressionPtr helper::Parser::makeExpression(const std::string &expr, const edm::TypeWithDict &type) {
  reco::parser::ExpressionPtr ret;

  using namespace boost::spirit::classic;
  reco::parser::Grammar grammar(ret, type, true);
  const char *startingFrom = expr.c_str();
  try {
    parse(startingFrom,
          grammar.use_parser<1>() >> end_p,
          space_p);  /// NOTA BENE: <0> for cuts, <1> for expresions (why? boh!)
  } catch (reco::parser::BaseException &e) {
    std::cerr << "Expression parser error:" << reco::parser::baseExceptionWhat(e) << " (char " << e.where - startingFrom
              << ")" << std::endl;
  }
  return ret;
}

reco::parser::SelectorPtr helper::Parser::makeSelector(const std::string &expr, const edm::TypeWithDict &type) {
  reco::parser::SelectorPtr ret;

  using namespace boost::spirit::classic;
  reco::parser::Grammar grammar(ret, type, true);
  const char *startingFrom = expr.c_str();
  try {
    parse(startingFrom,
          grammar.use_parser<0>() >> end_p,
          space_p);  /// NOTA BENE: <0> for cuts, <1> for expresions (why? boh!)
  } catch (reco::parser::BaseException &e) {
    std::cerr << "Selector parser error:" << reco::parser::baseExceptionWhat(e) << " (char " << e.where - startingFrom
              << ")" << std::endl;
  }
  return ret;
}

edm::TypeWithDict helper::Parser::elementType(const edm::TypeWithDict &wrapperType) {
  edm::TypeWithDict collection = wrapperType.templateArgumentAt(0);
  // now search for value_type
  edm::TypeWithDict objtype = collection.nestedType("value_type");
  if (bool(objtype)) {
    return objtype;
  }
  std::cerr << "Can't get a type out of " << wrapperType.name() << std::endl;
  return edm::TypeWithDict();
}

bool helper::Parser::test(const reco::parser::SelectorPtr &sel, const edm::TypeWithDict type, const void *ptr) {
  if (sel.get() == nullptr)
    return false;
  edm::ObjectWithDict obj(type, const_cast<void *>(ptr));
  return (*sel)(obj);
}

double helper::Parser::eval(const reco::parser::ExpressionPtr &expr, const edm::TypeWithDict type, const void *ptr) {
  if (expr.get() == nullptr)
    return 0;
  edm::ObjectWithDict obj(type, const_cast<void *>(ptr));
  return expr->value(obj);
}

bool helper::ScannerBase::addExpression(const char *expr) {
  bool ok = true;
  exprs_.push_back(helper::Parser::makeExpression(expr, objType_));
  if (exprs_.back().get() == nullptr) {
    std::cerr << "Failed to parse expression " << expr << std::endl;
    exprs_.pop_back();
    ok = false;
  }
  return ok;
}

bool helper::ScannerBase::setCut(const char *cut) {
  bool ok = true;
  cuts_[0] = helper::Parser::makeSelector(cut, objType_);
  if (strlen(cut) && !cuts_[0].get()) {
    std::cerr << "Failed to set cut \"" << cut << "\"" << std::endl;
    ok = false;
  }
  return ok;
}

void helper::ScannerBase::clearCut() { cuts_[0].reset(); }

void helper::ScannerBase::clearExtraCuts() { cuts_.resize(1); }

bool helper::ScannerBase::addExtraCut(const char *cut) {
  bool ok = true;
  cuts_.push_back(helper::Parser::makeSelector(cut, objType_));
  if (!cuts_.back().get()) {
    std::cerr << "Failed to add cut \"" << cut << "\"" << std::endl;
    ok = false;
    cuts_.pop_back();
  }
  return ok;
}

bool helper::ScannerBase::test(const void *ptr, size_t icut) const {
  if (icut >= cuts_.size())
    return false;
  if (cuts_[icut].get() == nullptr)
    return true;
  try {
    edm::ObjectWithDict obj(objType_, const_cast<void *>(ptr));
    return (*cuts_[icut])(obj);
  } catch (std::exception &ex) {
    if (!ignoreExceptions_)
      std::cerr << "Caught exception " << ex.what() << std::endl;
    return false;
  }
}

double helper::ScannerBase::eval(const void *ptr, size_t iexpr) const {
  try {
    edm::ObjectWithDict obj(objType_, const_cast<void *>(ptr));
    if (exprs_.size() > iexpr)
      return exprs_[iexpr]->value(obj);
  } catch (std::exception &ex) {
    if (!ignoreExceptions_)
      std::cerr << "Caught exception " << ex.what() << std::endl;
  }
  return 0;
}

void helper::ScannerBase::print(const void *ptr) const {
  edm::ObjectWithDict obj(objType_, const_cast<void *>(ptr));
  if ((cuts_[0].get() == nullptr) || (*cuts_[0])(obj)) {
    for (std::vector<reco::parser::ExpressionPtr>::const_iterator it = exprs_.begin(), ed = exprs_.end(); it != ed;
         ++it) {
      if (ptr == nullptr || it->get() == nullptr) {
        printf(" : %8s", "#ERR");
      } else {
        try {
          double val = (*it)->value(obj);
          // I found no easy ways to enforce a fixed width from printf that works also with leading zeroes or large exponents (e.g. 1e15 or 1e101)
          // So we have to go the ugly way
          char buff[255];
          int len = sprintf(buff, " : % 8.6g", val);  // this is usually ok, and should be 3+8 chars long
          if (len == 3 + 8) {
            std::cout << buff;
          } else {
            if (strchr(buff, 'e')) {
              printf((len == 3 + 13 ? " :  % .0e" : " : % .1e"), val);
            } else {
              printf("%11.11s", buff);
            }
          }
        } catch (std::exception &ex) {
          printf(" : %8s", "EXCEPT");
          if (!ignoreExceptions_)
            std::cerr << "Caught exception " << ex.what() << std::endl;
        }
      }
    }
    for (std::vector<reco::parser::SelectorPtr>::const_iterator it = cuts_.begin() + 1, ed = cuts_.end(); it != ed;
         ++it) {
      if (ptr == nullptr || it->get() == nullptr) {
        printf(" : %8s", "#ERR");
      } else {
        try {
          int ret = (*it)->operator()(obj);
          printf(" : %8d", ret);
        } catch (std::exception &ex) {
          printf(" : %8s", "EXCEPT");
          if (!ignoreExceptions_)
            std::cerr << "Caught exception " << ex.what() << std::endl;
        }
      }
    }
    fflush(stdout);
  }
}

void helper::ScannerBase::fill1D(const void *ptr, TH1 *hist) const {
  edm::ObjectWithDict obj(objType_, const_cast<void *>(ptr));
  if ((cuts_[0].get() == nullptr) || (*cuts_[0])(obj)) {
    try {
      if (!exprs_.empty())
        hist->Fill(exprs_[0]->value(obj));
    } catch (std::exception &ex) {
      if (!ignoreExceptions_)
        std::cerr << "Caught exception " << ex.what() << std::endl;
    }
  }
}

void helper::ScannerBase::fill2D(const void *ptr, TH2 *hist) const {
  edm::ObjectWithDict obj(objType_, const_cast<void *>(ptr));
  if ((cuts_[0].get() == nullptr) || (*cuts_[0])(obj)) {
    try {
      if (exprs_.size() >= 2)
        hist->Fill(exprs_[0]->value(obj), exprs_[1]->value(obj));
    } catch (std::exception &ex) {
      if (!ignoreExceptions_)
        std::cerr << "Caught exception " << ex.what() << std::endl;
    }
  }
}

void helper::ScannerBase::fillGraph(const void *ptr, TGraph *graph) const {
  edm::ObjectWithDict obj(objType_, const_cast<void *>(ptr));
  if ((cuts_[0].get() == nullptr) || (*cuts_[0])(obj)) {
    try {
      if (exprs_.size() >= 2)
        graph->SetPoint(graph->GetN(), exprs_[0]->value(obj), exprs_[1]->value(obj));
    } catch (std::exception &ex) {
      if (!ignoreExceptions_)
        std::cerr << "Caught exception " << ex.what() << std::endl;
    }
  }
}

void helper::ScannerBase::fillProf(const void *ptr, TProfile *hist) const {
  edm::ObjectWithDict obj(objType_, const_cast<void *>(ptr));
  if ((cuts_[0].get() == nullptr) || (*cuts_[0])(obj)) {
    try {
      if (exprs_.size() >= 2)
        hist->Fill(exprs_[0]->value(obj), exprs_[1]->value(obj));
    } catch (std::exception &ex) {
      if (!ignoreExceptions_)
        std::cerr << "Caught exception " << ex.what() << std::endl;
    }
  }
}
