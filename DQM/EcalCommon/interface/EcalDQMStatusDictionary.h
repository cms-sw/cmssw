#ifndef EcalDQMStatusDictionary_H
#define EcalDQMStatusDictionary_H

/*!
  \file Status.h
  \brief dictionary for Ecal DQM status codes
  \author G. Della Ricca
  \version $Revision: 1.15 $
  \date $Date: 2011/09/14 13:51:23 $
*/

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"
#include <cstdint>
#include <string>
#include <vector>
#include <map>

class EcalDQMStatusDictionary {

 public:
  static void init();
  static void clear();
  static uint32_t getCode(std::string const&);
  static std::vector<std::string> getNames(uint32_t);

 private:

  EcalDQMStatusDictionary() {}; // Hidden to force static use
  ~EcalDQMStatusDictionary() {}; // Hidden to force static use

  static std::map<std::string, uint32_t> codeMap;
};

#endif // EcalDQMStatusDictionary_H
