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
  for (LineMap::const_iterator it = m_lines.begin(); it != m_lines.end(); ++it) {
    out << "Event no: " << it->first.first << std::endl
        << "Bx no   : " << it->first.second << std::endl
        << "Patterns: ";

    it->second.print(out);

    out << std::endl;
  }
}
