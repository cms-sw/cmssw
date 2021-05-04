/**
 * \class L1GtPatternWriter
 * 
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
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtPatternWriter.h"

// system include files
#include <iostream>
#include <iomanip>
#include <algorithm>

// user include files
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtPatternMap.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtPatternWriter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

L1GtPatternWriter::L1GtPatternWriter(std::ostream& destination,
                                     const std::string& header,
                                     const std::string& footer,
                                     const std::vector<std::string>& columns,
                                     const std::vector<uint32_t>& lengths,
                                     const std::vector<uint32_t>& defaults,
                                     const std::vector<int>& bx,
                                     bool debug)
    : m_dest(destination),
      m_header(header),
      m_footer(footer),
      m_columns(columns),
      m_lengths(lengths),
      m_defaults(defaults),
      m_bx(bx),
      m_debug(debug),
      m_lineNo(0)

{
  m_dest << m_header;
}

void L1GtPatternWriter::writePatterns(const L1GtPatternMap& patterns) {
  for (L1GtPatternMap::LineMap::const_iterator it = patterns.begin(); it != patterns.end(); ++it) {
    int event = it->first.first;
    int bx = it->first.second;

    if (edm::isDebugEnabled() && m_debug) {
      std::stringstream dump;
      patterns.print(dump);
      LogTrace("L1GtPatternGenerator") << dump.str();
    }

    if (m_bx.empty() || std::find(m_bx.begin(), m_bx.end(), bx) != m_bx.end()) {
      if (m_debug) {
        m_dest << "# Event " << std::dec << event << ", bx " << bx << std::endl;
      }
      writePatternLine(it->second);
      ++m_lineNo;
    } else {
      LogTrace("L1GtPatternGenerator") << "Skipping event " << it->first.first << " bx " << it->first.second
                                       << " because of bx filter";
    }
  }
}

void L1GtPatternWriter::writePatternLine(const L1GtPatternLine& line) {
  m_dest << std::setfill('0');
  // open each line with a line number
  // the line number is in decimal, while everything else is hex.
  m_dest << std::dec << std::setw(4) << m_lineNo << ' ' << std::hex;

  for (uint32_t i = 0; i < m_columns.size(); ++i) {
    // space beween fields
    if (i)
      m_dest << ' ';

    // retrieve column value
    uint32_t value;
    if (line.has(m_columns[i])) {
      // value comes from data
      value = line.get(m_columns[i]);
    } else if (m_defaults.size() > i) {
      // value was specified as a config default.
      value = m_defaults[i];
    } else {
      // no default specified, set to 0
      value = 0;
    }
    uint32_t digits = (m_lengths[i] + 3) / 4;

    // write to file with configured length (truncating value if neccessary)
    m_dest << std::setw(digits) << (value & mask(m_lengths[i]));
  }

  // next line
  m_dest << std::endl;
}

void L1GtPatternWriter::close() {
  if (m_dest) {
    m_dest << m_footer;
  }
}

L1GtPatternWriter::~L1GtPatternWriter() { close(); }

uint32_t L1GtPatternWriter::mask(uint32_t length) {
  if (length < 32) {
    return ~((~0U) << length);
  } else {
    return ~0U;
  }
}
