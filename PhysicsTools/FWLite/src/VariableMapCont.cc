// -*- C++ -*-
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cassert>
#include <algorithm>
#include "PhysicsTools/FWLite/interface/VariableMapCont.h"

using namespace std;
using namespace optutl;

const int VariableMapCont::kDefaultInteger = 0;
const double VariableMapCont::kDefaultDouble = 0.;
const std::string VariableMapCont::kDefaultString = "";
const bool VariableMapCont::kDefaultBool = false;
const VariableMapCont::IVec VariableMapCont::kEmptyIVec;
const VariableMapCont::DVec VariableMapCont::kEmptyDVec;
const VariableMapCont::SVec VariableMapCont::kEmptySVec;

VariableMapCont::VariableMapCont() {}

VariableMapCont::OptionType VariableMapCont::hasVariable(string key) {
  lowercaseString(key);
  // Look through our maps to see if we've got it
  if (m_integerMap.end() != m_integerMap.find(key))
    return kInteger;
  if (m_doubleMap.end() != m_doubleMap.find(key))
    return kDouble;
  if (m_stringMap.end() != m_stringMap.find(key))
    return kString;
  if (m_boolMap.end() != m_boolMap.find(key))
    return kBool;
  if (m_integerVecMap.end() != m_integerVecMap.find(key))
    return kIntegerVector;
  if (m_doubleVecMap.end() != m_doubleVecMap.find(key))
    return kDoubleVector;
  if (m_stringVecMap.end() != m_stringVecMap.find(key))
    return kStringVector;
  // if we're here, the answer's no.
  return kNone;
}

void VariableMapCont::lowercaseString(string &arg) {
  // assumes 'toLower(ch)' modifies ch
  std::for_each(arg.begin(), arg.end(), VariableMapCont::toLower);
  // // assumes 'toLower(ch)' returns the lower case char
  // std::transform (arg.begin(), arg.end(), arg.begin(),
  //                 VariableMapCont::toLower);
}

char VariableMapCont::toLower(char &ch) {
  ch = tolower(ch);
  return ch;
}

void VariableMapCont::_checkKey(string &key, const string &description) {
  // Let's make sure we don't already have this key
  lowercaseString(key);
  if (m_variableModifiedMap.end() != m_variableModifiedMap.find(key)) {
    cerr << "VariableMapCont::addVariable() Error: Key '" << key << "' has already been defined.  Aborting." << endl;
    assert(0);
  }  // found a duplicate
  m_variableModifiedMap[key] = false;
  m_variableDescriptionMap[key] = description;
}

void VariableMapCont::addOption(string key, OptionType type, const string &description) {
  _checkKey(key, description);
  if (kInteger == type) {
    m_integerMap[key] = kDefaultInteger;
    return;
  }
  if (kDouble == type) {
    m_doubleMap[key] = kDefaultDouble;
    return;
  }
  if (kString == type) {
    m_stringMap[key] = kDefaultString;
    return;
  }
  if (kBool == type) {
    m_boolMap[key] = kDefaultBool;
    return;
  }
  if (kIntegerVector == type) {
    m_integerVecMap[key] = kEmptyIVec;
    return;
  }
  if (kDoubleVector == type) {
    m_doubleVecMap[key] = kEmptyDVec;
    return;
  }
  if (kStringVector == type) {
    m_stringVecMap[key] = kEmptySVec;
    return;
  }
}

void VariableMapCont::addOption(string key, OptionType type, const string &description, int defaultValue) {
  _checkKey(key, description);
  if (kInteger != type) {
    cerr << "VariableMapCont::addOption() Error: Key '" << key << "' is not defined as an integer but has an integer "
         << "default value. Aborting." << endl;
    assert(0);
  }
  m_integerMap[key] = defaultValue;
}

void VariableMapCont::addOption(string key, OptionType type, const string &description, double defaultValue) {
  _checkKey(key, description);
  if (kDouble != type) {
    cerr << "VariableMapCont::addOption() Error: Key '" << key << "' is not defined as an double but has an double "
         << "default value. Aborting." << endl;
    assert(0);
  }
  m_doubleMap[key] = defaultValue;
}

void VariableMapCont::addOption(string key, OptionType type, const string &description, const string &defaultValue) {
  _checkKey(key, description);
  if (kString != type) {
    cerr << "VariableMapCont::addOption() Error: Key '" << key << "' is not defined as an string but has an string "
         << "default value. Aborting." << endl;
    assert(0);
  }
  m_stringMap[key] = defaultValue;
}

void VariableMapCont::addOption(string key, OptionType type, const string &description, const char *defaultValue) {
  addOption(key, type, description, (string)defaultValue);
}

void VariableMapCont::addOption(string key, OptionType type, const string &description, bool defaultValue) {
  _checkKey(key, description);
  if (kBool != type) {
    cerr << "VariableMapCont::addOption() Error: Key '" << key << "' is not defined as an bool but has an bool "
         << "default value. Aborting." << endl;
    assert(0);
  }
  m_boolMap[key] = defaultValue;
}

int &VariableMapCont::integerValue(std::string key) {
  lowercaseString(key);
  SIMapIter iter = m_integerMap.find(key);
  if (m_integerMap.end() == iter) {
    cerr << "VariableMapCont::integerValue() Error: key '" << key << "' not found.  Aborting." << endl;
    assert(0);
  }
  return iter->second;
}

double &VariableMapCont::doubleValue(std::string key) {
  lowercaseString(key);
  SDMapIter iter = m_doubleMap.find(key);
  if (m_doubleMap.end() == iter) {
    cerr << "VariableMapCont::doubleValue() Error: key '" << key << "' not found.  Aborting." << endl;
    assert(0);
  }
  return iter->second;
}

string &VariableMapCont::stringValue(std::string key) {
  lowercaseString(key);
  SSMapIter iter = m_stringMap.find(key);
  if (m_stringMap.end() == iter) {
    cerr << "VariableMapCont::stringValue() Error: key '" << key << "' not found.  Aborting." << endl;
    assert(0);
  }
  return iter->second;
}

bool &VariableMapCont::boolValue(std::string key) {
  lowercaseString(key);
  SBMapIter iter = m_boolMap.find(key);
  if (m_boolMap.end() == iter) {
    cerr << "VariableMapCont::boolValue() Error: key '" << key << "' not found.  Aborting." << endl;
    assert(0);
  }
  return iter->second;
}

VariableMapCont::IVec &VariableMapCont::integerVector(std::string key) {
  lowercaseString(key);
  SIVecMapIter iter = m_integerVecMap.find(key);
  if (m_integerVecMap.end() == iter) {
    cerr << "VariableMapCont::integerVector() Error: key '" << key << "' not found.  Aborting." << endl;
    assert(0);
  }
  return iter->second;
}

VariableMapCont::DVec &VariableMapCont::doubleVector(std::string key) {
  lowercaseString(key);
  SDVecMapIter iter = m_doubleVecMap.find(key);
  if (m_doubleVecMap.end() == iter) {
    cerr << "VariableMapCont::doubleVector() Error: key '" << key << "' not found.  Aborting." << endl;
    assert(0);
  }
  return iter->second;
}

VariableMapCont::SVec &VariableMapCont::stringVector(std::string key) {
  lowercaseString(key);
  SSVecMapIter iter = m_stringVecMap.find(key);
  if (m_stringVecMap.end() == iter) {
    cerr << "VariableMapCont::stringVector() Error: key '" << key << "' not found.  Aborting." << endl;
    assert(0);
  }
  return iter->second;
}

bool VariableMapCont::_valueHasBeenModified(const string &key) {
  SBMapConstIter iter = m_variableModifiedMap.find(key);
  if (m_variableModifiedMap.end() == iter) {
    // Not found.  Not a valid option
    cerr << "VariableMapCont::valueHasBeenModfied () Error: '" << key << "' is not a valid key." << endl;
    return false;
  }
  return iter->second;
}

// friends
ostream &operator<<(ostream &o_stream, const VariableMapCont &rhs) { return o_stream; }
