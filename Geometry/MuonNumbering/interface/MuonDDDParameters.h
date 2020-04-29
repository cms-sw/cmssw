#ifndef Geometry_MuonNumbering_MuonDDDParameters_h
#define Geometry_MuonNumbering_MuonDDDParameters_h

/** \class MuonDDDParameters
 *
 * this class reads the constant section of
 * the muon-numbering xml-file
 *  
 * \author Sunanda Banerjee
 *  modified by:
 *   Taken from MuonDDDConstants
 *
 */

#include <string>
#include <map>
#include <iostream>
#include "CondFormats/Serialization/interface/Serializable.h"

class MuonDDDParameters {
public:
  MuonDDDParameters() {}

  int getValue(const std::string& name) const;
  void addValue(const std::string& name, const int& value);
  unsigned size() const { return namesAndValues_.size(); }
  std::pair<std::string, int> getEntry(const unsigned int k) const {
    auto itr = namesAndValues_.begin();
    for (unsigned int i = 0; i < k; ++i)
      ++itr;
    if (k < size())
      return std::pair<std::string, int>(itr->first, itr->second);
    else
      return std::pair<std::string, int>("Not Found", 0);
  }

private:
  std::map<std::string, int> namesAndValues_;
};

#endif
