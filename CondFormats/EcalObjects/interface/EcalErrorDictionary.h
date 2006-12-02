#ifndef CondFormats_EcalObjects_EcalErrorDictionary_H
#define CondFormats_EcalObjects_EcalErrorDictionary_H

/**
 *  A dictionary of bitmasks for ECAL channel errors and their meaning
 *  This object is not meant to be stored in the offline DB, but the bits
 *  defined here are stored in EcalChannelStatus.
 *
 *  This class holds no dynamic data and all the methods are static.
 */
#include <boost/cstdint.hpp>
#include <iostream>
#include <vector>

class EcalErrorDictionary {
 public:
  struct errorDef_t {
    uint64_t bitmask;
    char shortDesc[64];
    char longDesc[128];
  };

  static uint64_t hasError(std::string shortDesc, uint64_t bitcode)
    {
      return getMask(shortDesc) & bitcode;
    }

  static uint64_t getMask(std::string shortDesc)
    {
      for (unsigned int i=0; i<DICTSIZE; i++) {
	if (getDef(i).shortDesc == shortDesc) {
	  return getDef(i).bitmask;
	}
      }
      return 0;
    }
  
  static void printErrors(uint64_t bitcode)
    {
      for (unsigned int i=0; i<DICTSIZE; i++) {
	if (bitcode & getDef(i).bitmask) {
	  std::cout << getDef(i).shortDesc << std::endl;
	}
      }
    }

  static void getErrors(std::vector<errorDef_t> errorVec, uint64_t bitcode)
    {
      errorVec.clear();
      for (unsigned int i=0; i<DICTSIZE; i++) {
	if (bitcode & getDef(i).bitmask) {
	  errorVec.push_back(getDef(i));
	}
      }
    }

  static void getDictionary(std::vector<errorDef_t> &dict)
    {
      dict.clear();
      for (unsigned int i=0; i<DICTSIZE; i++) {
	dict.push_back(getDef(i));
      }
    }

 private:
  EcalErrorDictionary() {}; // Hidden to force static use
  ~EcalErrorDictionary() {};  // Hidden to force static use

  const static unsigned int DICTSIZE = 4;
  
  static errorDef_t getDef(unsigned int i) {
    const static errorDef_t ERRORDICT[DICTSIZE] =
      {
	{ (1<<0), "PEDESTAL_MEAN_AMPLITUDE_TOO_LOW"  , "Pedestal mean amplitude too low" },
	{ (1<<1), "PEDESTAL_MEAN_AMPLITUDE_TOO_HIGH" , "Pedestal mean amplitude too high" },
	{ (1<<2), "PEDESTAL_RMS_AMPLITUDE_TOO_LOW"   , "Pedestal RMS amplitude too low" },
	{ (1<<3), "PEDESTAL_RMS_AMPLITUDE_TOO_HIGH"  , "Pedestal RMS amplitude too high"}
      };

    return ERRORDICT[i];
  }
};
#endif
