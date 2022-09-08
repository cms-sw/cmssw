#include <iostream>
#include <regex>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/RegexMatch.h"
#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionData.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionL1uGTReader.h"

namespace triggerExpression {

  // define the result of the module from the L1 reults
  bool L1uGTReader::operator()(const Data& data) const {
    if (not data.hasL1T())
      return false;

    std::vector<bool> const& word = data.l1tResults();
    if (word.empty())
      return false;

    auto const& triggers = masksEnabled() ? m_triggersAfterMasking : m_triggers;

    for (auto const& trigger : triggers)
      if (trigger.second < word.size() and word[trigger.second])
        return true;

    return false;
  }

  void L1uGTReader::dump(std::ostream& out, bool const ignoreMasks) const {
    if (not m_initialised) {
      out << "Uninitialised_L1_Expression";
      return;
    }

    auto const& triggers = ignoreMasks or not masksEnabled() ? m_triggers : m_triggersAfterMasking;

    if (triggers.empty()) {
      out << "FALSE";
    } else if (triggers.size() == 1) {
      out << triggers[0].first;
    } else {
      out << "(" << triggers[0].first;
      for (unsigned int i = 1; i < triggers.size(); ++i)
        out << " OR " << triggers[i].first;
      out << ")";
    }
  }

  void L1uGTReader::init(const Data& data) {
    if (not data.hasL1T())
      return;

    const L1TUtmTriggerMenu& menu = data.l1tMenu();

    // clear the previous configuration
    m_triggers.clear();

    // check if the pattern has is a glob expression, or a single trigger name
    auto const& triggerMap = menu.getAlgorithmMap();
    if (not edm::is_glob(m_pattern)) {
      // no wildcard expression
      auto entry = triggerMap.find(m_pattern);
      if (entry != triggerMap.end()) {
        // single L1 bit
        m_triggers.push_back(std::make_pair(m_pattern, entry->second.getIndex()));
      } else {
        // trigger not found in the current menu
        if (data.shouldThrow())
          throw cms::Exception("Configuration")
              << "requested L1 trigger \"" << m_pattern << "\" does not exist in the current L1 menu";
        else
          edm::LogWarning("Configuration")
              << "requested L1 trigger \"" << m_pattern << "\" does not exist in the current L1 menu";
      }
    } else {
      // expand wildcards in the pattern
      bool match = false;
      std::regex re(edm::glob2reg(m_pattern));
      for (auto const& entry : triggerMap)
        if (std::regex_match(entry.first, re)) {
          match = true;
          m_triggers.push_back(std::make_pair(entry.first, entry.second.getIndex()));
        }

      if (not match) {
        // m_pattern does not match any L1 bits
        if (data.shouldThrow())
          throw cms::Exception("Configuration")
              << "requested pattern \"" << m_pattern << "\" does not match any L1 trigger in the current menu";
        else
          edm::LogWarning("Configuration")
              << "requested pattern \"" << m_pattern << "\" does not match any L1 trigger in the current menu";
      }
    }

    m_triggersAfterMasking = m_triggers;
    m_initialised = true;
  }

  void L1uGTReader::mask(Evaluator const& eval) {
    auto const& triggersToMask = eval.triggers();

    if (triggersToMask.empty()) {
      edm::LogInfo("NoTriggersToMask") << "\tL1uGTReader[\"" << *this << "\"]::mask(arg = \"" << eval << "\")"
                                       << " failed: arg.triggers() is empty";
      return;
    }

    // patterns() is always empty for a L1uGTReader, and not empty for PathReader;
    // here, PathReader evaluators are skipped as they shouldn't be used to mask a L1uGTReader
    if (not eval.patterns().empty()) {
      edm::LogWarning("InvalidArgumentForMasking")
          << "\tL1uGTReader[\"" << *this << "\"]::mask(arg = \"" << eval << "\")"
          << " failed: arg.patterns() is not empty (arg is not a L1uGTReader)";
      return;
    }

    enableMasks();

    // clang-format off
    m_triggersAfterMasking.erase(
      std::remove_if(
        m_triggersAfterMasking.begin(),
        m_triggersAfterMasking.end(),
        [&triggersToMask](auto const& foo) {
          return std::find(triggersToMask.begin(), triggersToMask.end(), foo) != triggersToMask.end();
        }
      ),
      m_triggersAfterMasking.end()
    );
    // clang-format on
  }

}  // namespace triggerExpression
