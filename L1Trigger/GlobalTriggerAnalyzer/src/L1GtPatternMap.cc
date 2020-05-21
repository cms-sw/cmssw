/**
 * \class L1GtPatternMap
 * 
 * Description: see header file.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Thomas Themel - HEPHY Vienna
 * 
 *
 */

// this class header
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtPatternMap.h"

#include <iostream>
#include <sstream>

L1GtPatternLine& L1GtPatternMap::getLine(int eventNr, int bxNr) {
  std::pair<int, int> key = std::make_pair(eventNr, bxNr);
  return m_lines[key];
}

L1GtPatternMap::LineMap::const_iterator L1GtPatternMap::begin() const { return m_lines.begin(); }
L1GtPatternMap::LineMap::const_iterator L1GtPatternMap::end() const { return m_lines.end(); }

L1GtPatternMap::LineMap::iterator L1GtPatternMap::begin() { return m_lines.begin(); }
L1GtPatternMap::LineMap::iterator L1GtPatternMap::end() { return m_lines.end(); }

void L1GtPatternMap::print(std::ostream& out) const {
  for (const auto& m_line : m_lines) {
    out << "Event no: " << m_line.first.first << std::endl
        << "Bx no   : " << m_line.first.second << std::endl
        << "Patterns: ";

    m_line.second.print(out);

    out << std::endl;
  }
}
