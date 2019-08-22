#ifndef GlobalTriggerAnalyzer_L1GtPatternMap_h
#define GlobalTriggerAnalyzer_L1GtPatternMap_h

/**
 * \class L1GtPatternMap
 * 
 * 
 * Description: A class to map events to their proper pattern file lines
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Thomas Themel - HEPHY Vienna
 * 
 *
 */

#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtPatternLine.h"

/** The purpose of the L1GtPatternMap is to split input into pattern file lines
    identified by event/bunch crossing number. */
class L1GtPatternMap {
public:
  /** Returns the pattern line for a certain event/bx combination,
     creating it if neccessary. */
  L1GtPatternLine& getLine(int eventNr, int bxNr);

  typedef std::map<std::pair<int, int>, L1GtPatternLine> LineMap;

  /** Export iteration support. */
  LineMap::const_iterator begin() const;
  LineMap::const_iterator end() const;

  /** Export iteration support. */
  LineMap::iterator begin();
  LineMap::iterator end();

  /** Debug dump. */
  void print(std::ostream& out) const;

private:
  LineMap m_lines;
};

#endif /*GlobalTriggerAnalyzer_L1GtPatternMap_h*/
