#include "TriggerSelector.h"

#include "DataFormats/Common/interface/HLTPathStatus.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/EventSelector.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/RegexMatch.h"

#include <iostream>
#include <algorithm>

#include <boost/regex.hpp>

namespace dqmservices {

  // compatibility constructor

  TriggerSelector::TriggerSelector(Strings const& pathspecs, Strings const& triggernames) : useOld_(true) {
    acceptAll_ = false;
    eventSelector_.reset(new edm::EventSelector(pathspecs, triggernames));
  }

  TriggerSelector::TriggerSelector(std::string const& expression, Strings const& triggernames) : useOld_(false) {
    init(trim(expression), triggernames);
  }

  void TriggerSelector::init(std::string const& expression, Strings const& triggernames) {
    // debug_ = true;
    if (expression.empty()) {
      acceptAll_ = true;
      return;
    }
    if (expression.size() == 1 && expression.at(0) == '*')
      acceptAll_ = true;
    else
      acceptAll_ = false;

    // replace all possible alternate operators (.AND. and .OR.)
    {
      using namespace boost;
      std::string temp;
      temp = regex_replace(expression, regex(".AND."), "&&");
      expression_ = regex_replace(temp, regex(".and."), "&&");
      temp = regex_replace(expression_, regex(".OR."), "||");
      expression_ = regex_replace(temp, regex(".or."), "||");
    }

    // build decision tree
    masterElement_.reset(new TreeElement(expression_, triggernames));
  }

  bool TriggerSelector::acceptEvent(edm::TriggerResults const& tr) const {
    if (useOld_) {
      return eventSelector_->acceptEvent(tr);
    }

    if (acceptAll_)
      return true;

    return masterElement_->returnStatus(tr);
  }

  bool TriggerSelector::acceptEvent(unsigned char const* array_of_trigger_results, int number_of_trigger_paths) const {
    if (useOld_)
      return eventSelector_->acceptEvent(array_of_trigger_results, number_of_trigger_paths);

    if (acceptAll_)
      return true;

    // Form HLTGlobalStatus object to represent the array_of_trigger_results
    edm::HLTGlobalStatus tr(number_of_trigger_paths);
    int byteIndex = 0;
    int subIndex = 0;
    for (int pathIndex = 0; pathIndex < number_of_trigger_paths; ++pathIndex) {
      int state = array_of_trigger_results[byteIndex] >> (subIndex * 2);
      state &= 0x3;
      edm::HLTPathStatus pathStatus(static_cast<edm::hlt::HLTState>(state));
      tr[pathIndex] = pathStatus;
      ++subIndex;
      if (subIndex == 4) {
        ++byteIndex;
        subIndex = 0;
      }
    }
    // Now make the decision, based on the HLTGlobalStatus tr,
    // which we have created from the supplied array of results
    masterElement_->returnStatus(tr);
    return masterElement_->returnStatus(tr);
  }

  TriggerSelector::TreeElement::TreeElement(std::string const& inputString,
                                            Strings const& tr,
                                            TreeElement* parentElement)
      : op_(NonInit), trigBit_(-1) {
    std::string str_ = trim(inputString);
    children_.clear();
    parent_ = parentElement;

    size_t offset_ = 0;
    bool occurrences_ = false;

    if (str_.empty())
      throw edm::Exception(edm::errors::Configuration) << "Syntax Error (empty element)";

    static const size_t bopsSize_ = 2;
    static const std::string binaryOperators_[bopsSize_] = {"||", "&&"};

    for (size_t opr = 0; opr < bopsSize_; opr++) {
      bool exitloop_ = false;
      while (!exitloop_) {
        size_t t_end_;

        std::string tmpStr = str_.substr(offset_);
        t_end_ = tmpStr.find(binaryOperators_[opr]);
        if (debug_)
          std::cout << "offset: " << offset_ << " length: " << t_end_ << " string: " << tmpStr << std::endl;

        if (t_end_ == std::string::npos) {
          // right side element
          if (occurrences_)
            children_.push_back(new TreeElement(tmpStr, tr, this));
          break;
        }
        t_end_ += offset_;
        if (t_end_ == 0 || t_end_ + 2 >= str_.size())
          throw edm::Exception(edm::errors::Configuration) << "Syntax Error (operator is not unary)\n";
        else {
          // count bracket in preceeding part
          size_t brackets_ = 0;
          for (size_t k = offset_; k < t_end_; k++) {
            if (str_.at(k) == '(') {
              brackets_++;
            } else if (str_.at(k) == ')') {
              if (brackets_ == 0) {
                throw edm::Exception(edm::errors::Configuration) << "Syntax Error (brackets)\n";
              } else {
                brackets_--;
              }
            }
          }
          if (brackets_ == 0) {
            std::string next = str_.substr(offset_, t_end_ - offset_);
            children_.push_back(new TreeElement(next, tr, this));
            occurrences_ = true;
            offset_ = t_end_ + 2;
          } else {
            // operator is inside brackets, find another
            int bracketcnt_ = 0;
            for (size_t k = offset_; true; k++) {
              if (k >= str_.size()) {
                if (bracketcnt_ != 0)
                  throw edm::Exception(edm::errors::Configuration) << "Syntax Error (brackets)\n";
                exitloop_ = true;
                if (occurrences_) {
                  children_.push_back(new TreeElement(str_.substr(offset_), tr, this));
                }
                break;
              }
              // look for another operator
              if (k >= t_end_ + 2 && bracketcnt_ == 0) {
                std::string temp = str_.substr(k);
                size_t pos = temp.find(binaryOperators_[opr]);
                if (pos == std::string::npos) {
                  exitloop_ = true;
                  if (occurrences_) {
                    children_.push_back(new TreeElement(str_.substr(offset_), tr, this));
                  }
                  break;
                } else {
                  int brcount_ = 0;
                  for (size_t s = 0; s < pos; s++) {
                    // counting check of brackets from last position to operator
                    if (temp.at(pos) == '(') {
                      brcount_++;
                    } else if (temp.at(pos) == ')') {
                      if (brcount_ == 0) {
                        throw edm::Exception(edm::errors::Configuration) << "Syntax error (brackets)\n";
                      } else {
                        brcount_--;
                      }
                    }
                  }
                  if (brcount_ != 0)
                    throw edm::Exception(edm::errors::Configuration) << "Syntax error (brackets)\n";

                  children_.push_back(new TreeElement(str_.substr(offset_, pos + k), tr, this));
                  offset_ = k + pos + 2;
                  occurrences_ = true;
                  if (offset_ >= str_.size())
                    throw edm::Exception(edm::errors::Configuration) << "Syntax Error (operator is not unary)\n";
                  break;
                }
              }

              if (str_.at(k) == '(')
                bracketcnt_++;
              if (str_.at(k) == ')')
                bracketcnt_--;
            }
          }
        }
      }
      if (occurrences_) {
        if (opr == 0)
          op_ = OR;
        else
          op_ = AND;
        return;
      }
    }

    if (str_.empty()) {
      op_ = AND;
      if (debug_)
        std::cout << "warning: empty element (will return true)" << std::endl;
      return;
    }

    if (str_.at(0) == '!') {
      op_ = NOT;
      std::string next = str_.substr(1);
      children_.push_back(new TreeElement(next, tr, this));
      return;
    }
    size_t beginBlock_ = str_.find('(');
    size_t endBlock_ = str_.rfind(')');
    bool found_lbracket = (beginBlock_ != std::string::npos);
    bool found_rbracket = (endBlock_ != std::string::npos);

    if (found_lbracket != found_rbracket) {
      throw edm::Exception(edm::errors::Configuration) << "Syntax Error (brackets)\n";
    } else if (found_lbracket && found_rbracket) {
      if (beginBlock_ >= endBlock_) {
        throw edm::Exception(edm::errors::Configuration) << "Syntax Error (brackets)\n";
      }
      if (beginBlock_ != 0 || endBlock_ != str_.size() - 1)
        throw edm::Exception(edm::errors::Configuration) << "Syntax Error (invalid character)\n";

      std::string next = str_.substr(beginBlock_ + 1, endBlock_ - beginBlock_ - 1);

      children_.push_back(new TreeElement(next, tr, this));
      op_ = BR;  // a bracket
      return;
    } else if (!found_lbracket && !found_rbracket)  // assume single trigger or wildcard (parsing)
    {
      bool ignore_if_missing = true;
      size_t chr_pos = str_.find('@');
      if (chr_pos != std::string::npos) {
        ignore_if_missing = false;
        str_ = str_.substr(0, chr_pos);
      }

      std::vector<Strings::const_iterator> matches = edm::regexMatch(tr, str_);
      if (matches.empty()) {
        if (!ignore_if_missing)  // && !edm::is_glob(str_))
          throw edm::Exception(edm::errors::Configuration) << "Trigger name (or match) not present";
        else {
          if (debug_)
            std::cout << "TriggerSelector: Couldn't match any triggers from: " << str_ << std::endl
                      << "                 Node will not be added " << std::endl;
          op_ = OR;
          return;
        }
      }
      if (matches.size() == 1) {
        // Single Trigger match
        trigBit_ = distance(tr.begin(), matches[0]);
        if (debug_)
          std::cout << "added trigger path: " << trigBit_ << std::endl;
        return;
      }
      if (matches.size() > 1) {
        op_ = OR;
        for (size_t l = 0; l < matches.size(); l++)
          children_.push_back(new TreeElement(*(matches[l]), tr, this));
      }
    }
  }

  std::string TriggerSelector::trim(std::string input) {
    if (!input.empty()) {
      std::string::size_type pos = input.find_first_not_of(' ');
      if (pos != std::string::npos)
        input.erase(0, pos);

      pos = input.find_last_not_of(' ');
      if (pos != std::string::npos)
        input.erase(pos + 1);
    }
    return input;
  }

  std::string TriggerSelector::makeXMLString(std::string const& input) {
    std::string output;
    if (!input.empty()) {
      for (size_t pos = 0; pos < input.size(); pos++) {
        char ch = input.at(pos);
        if (ch == '&')
          output.append("&amp;");
        else
          output.append(1, ch);
      }
    }
    return output;
  }

  bool TriggerSelector::TreeElement::returnStatus(edm::HLTGlobalStatus const& trStatus) const {
    if (children_.empty()) {
      if (op_ == OR || op_ == NOT)
        return false;
      if (op_ == AND || op_ == BR)
        return true;

      if (trigBit_ < 0 || (unsigned int)trigBit_ >= trStatus.size())
        throw edm::Exception(edm::errors::Configuration) << "Internal Error: array out of bounds.";

      if ((trStatus[trigBit_]).state() == edm::hlt::Pass)
        return true;
      // else if ((trStatus[trigBit]).state() == edm::hlt::Fail) return false;

      return false;
    }
    if (op_ == NOT) {  // NEGATION
      return !children_[0]->returnStatus(trStatus);
    }
    if (op_ == BR) {  // BRACKET
      return children_[0]->returnStatus(trStatus);
    }
    if (op_ == AND) {  // AND
      bool status = true;
      for (size_t i = 0; i < children_.size(); i++)
        status = status && children_[i]->returnStatus(trStatus);
      return status;
    } else if (op_ == OR) {  // OR
      bool status = false;
      for (size_t i = 0; i < children_.size(); i++)
        status = status || children_[i]->returnStatus(trStatus);
      return status;
    }
    throw edm::Exception(edm::errors::Configuration)
        << "Internal error: reached end of returnStatus(...), op:state = " << op_;
    return false;
  }

  TriggerSelector::TreeElement::~TreeElement() {
    for (std::vector<TreeElement*>::iterator it = children_.begin(); it != children_.end(); it++)
      delete *it;
    children_.clear();
  }
}  // namespace dqmservices
