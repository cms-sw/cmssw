/**
 * \class L1GtPatternLine
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
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtPatternLine.h"

// system include files
#include <sstream>
#include <iostream>

// user include files
#include "FWCore/Utilities/interface/Exception.h"

void L1GtPatternLine::push(const std::string& prefix, uint32_t value) {
  std::string colName = nextName(prefix);
  // add three values for each column - the full 32bit value,
  // one for the lower 16 bits and one for the higher 16 bits.
  m_columns[colName] = value;
  m_columns[colName + "_h"] = value >> 16;
  m_columns[colName + "_l"] = value & 0xFFFF;
}

void L1GtPatternLine::set(const std::string& name, uint32_t value) {
  ColumnMap::iterator it = m_columns.find(name);
  if (it == m_columns.end()) {
    throw cms::Exception(__func__) << "Can't set field " << name << " to " << std::hex << value << ": not found";
  }

  it->second = value;
  m_columns[name + "_h"] = value >> 16;
  m_columns[name + "_l"] = value & 0xFFFF;
}

void L1GtPatternLine::print(std::ostream& out) const {
  out << "BEGIN Columns: " << std::endl;
  for (L1GtPatternLine::ColumnMap::const_iterator it = m_columns.begin(); it != m_columns.end(); ++it) {
    out << it->first << ": " << std::hex << it->second << std::endl;
  }
  out << "END Columns." << std::endl;
}

bool L1GtPatternLine::has(const std::string& colname) const { return m_columns.find(colname) != m_columns.end(); }

std::string L1GtPatternLine::nextName(const std::string& prefix) {
  int i = 1;
  std::string result;
  do {
    result = name(prefix, i++);
  } while (has(result));

  return result;
}

std::string L1GtPatternLine::name(const std::string& prefix, unsigned int i) const {
  std::ostringstream ss;
  ss << prefix << i;
  return ss.str();
}

uint32_t L1GtPatternLine::get(const std::string& name) const {
  ColumnMap::const_iterator it = m_columns.find(name);
  if (it != m_columns.end()) {
    return it->second;
  }
  return 0;
}
