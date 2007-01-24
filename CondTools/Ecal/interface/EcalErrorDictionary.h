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

  static void getErrors(std::vector<errorDef_t>& errorVec, uint64_t bitcode)
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

  const static unsigned int DICTSIZE = 40;
  
  static errorDef_t getDef(unsigned int i) {
    const static errorDef_t ERRORDICT[DICTSIZE] =
      {

	{ ((uint64_t)1<<0) , "PEDESTAL_LOW_GAIN_MEAN_WARNING" , "Pedestal low gain mean amplitude outside range"},
	{ ((uint64_t)1<<1) , "PEDESTAL_MIDDLE_GAIN_MEAN_WARNING" , "Pedestal middle gain mean amplitude outside range"},
	{ ((uint64_t)1<<2) , "PEDESTAL_HIGH_GAIN_MEAN_WARNING" , "Pedestal high gain mean amplitude outside range"},
	{ ((uint64_t)1<<3) , "PEDESTAL_LOW_GAIN_MEAN_ERROR" , "Pedestal low gain mean amplitude error"},
	{ ((uint64_t)1<<4) , "PEDESTAL_MIDDLE_GAIN_MEAN_ERROR" , "Pedestal middle gain mean amplitude error"},
	{ ((uint64_t)1<<5) , "PEDESTAL_HIGH_GAIN_MEAN_ERROR" , "Pedestal high gain mean amplitude error"},

	{ ((uint64_t)1<<6) , "PEDESTAL_LOW_GAIN_RMS_WARNING" , "Pedestal low gain rms amplitude outside range"},
	{ ((uint64_t)1<<7) , "PEDESTAL_MIDDLE_GAIN_RMS_WARNING" , "Pedestal middle gain rms amplitude outside range"},
	{ ((uint64_t)1<<8) , "PEDESTAL_HIGH_GAIN_RMS_WARNING" , "Pedestal high gain rms amplitude outside range"},
	{ ((uint64_t)1<<9) , "PEDESTAL_LOW_GAIN_RMS_ERROR" , "Pedestal low gain rms amplitude error"},
	{ ((uint64_t)1<<10), "PEDESTAL_MIDDLE_GAIN_RMS_ERROR" , "Pedestal middle gain rms amplitude error"},
	{ ((uint64_t)1<<11), "PEDESTAL_HIGH_GAIN_RMS_ERROR" , "Pedestal high gain rms amplitude error"},

	{ ((uint64_t)1<<12), "PEDESTAL_ONLINE_HIGH_GAIN_MEAN_WARNING" , "Pedestal online high gain mean amplitude outside range"},
	{ ((uint64_t)1<<13), "PEDESTAL_ONLINE_HIGH_GAIN_RMS_WARNING" , "Pedestal online high gain rms amplitude outside range"},
	{ ((uint64_t)1<<14), "PEDESTAL_ONLINE_HIGH_GAIN_MEAN_ERROR" ,  "Pedestal online high gain mean amplitude error"},
	{ ((uint64_t)1<<15), "PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR" , "Pedestal online high gain rms amplitude error"},

	{ ((uint64_t)1<<16), "TESTPULSE_LOW_GAIN_MEAN_WARNING" , "Testpulse low gain mean amplitude outside range"},
	{ ((uint64_t)1<<17), "TESTPULSE_MIDDLE_GAIN_MEAN_WARNING" , "Testpulse middle gain mean amplitude outside range"},
	{ ((uint64_t)1<<18), "TESTPULSE_HIGH_GAIN_MEAN_WARNING" , "Testpulse high gain mean amplitude outside range"},
	{ ((uint64_t)1<<19), "TESTPULSE_LOW_GAIN_RMS_WARNING" , "Testpulse low gain rms amplitude outside range"},
	{ ((uint64_t)1<<20), "TESTPULSE_MIDDLE_GAIN_RMS_WARNING" , "Testpulse middle gain rms amplitude outside range"},
	{ ((uint64_t)1<<21), "TESTPULSE_HIGH_GAIN_RMS_WARNING" , "Testpulse high gain rms amplitude outside range"},

	{ ((uint64_t)1<<22), "LASER_MEAN_WARNING" , "Laser mean amplitude outside range"},
	{ ((uint64_t)1<<23), "LASER_RMS_WARNING" , "Laser rms amplitude outside range"},

	{ ((uint64_t)1<<24), "LASER_MEAN_OVER_PN_WARNING" , "Laser mean amplitude over PN outside range"},
	{ ((uint64_t)1<<25), "LASER_RMS_OVER_PN_WARNING" , "Laser rms amplitude over PN outside range"},

	{ ((uint64_t)1<<32), "CH_ID_WARNING" , "Channel id warning"},
	{ ((uint64_t)1<<33), "CH_GAIN_ZERO_WARNING" , "Channel gain zero warning"},
	{ ((uint64_t)1<<34), "CH_GAIN_SWITCH_WARNING" , "Channel gain switch warning"},
	{ ((uint64_t)1<<35), "CH_ID_ERROR" , "Channel id error"},
	{ ((uint64_t)1<<36), "CH_GAIN_ZERO_ERROR" , "Channel gain zero error"},
	{ ((uint64_t)1<<37), "CH_GAIN_SWITCH_ERROR" , "Channel gain switch error"},

	{ ((uint64_t)1<<38), "TT_ID_WARNING" , "TT id warning"},
	{ ((uint64_t)1<<39), "TT_SIZE_WARNING" , "TT size warning"},
	{ ((uint64_t)1<<40), "TT_LV1_WARNING" , "TT LV1 warning"},
	{ ((uint64_t)1<<41), "TT_BUNCH_X_WARNING" , "TT bunch-x warning"},
	{ ((uint64_t)1<<42), "TT_ID_ERROR" , "TT id error"},
	{ ((uint64_t)1<<43), "TT_SIZE_ERROR" , "TT size error"},
	{ ((uint64_t)1<<44), "TT_LV1_ERROR" , "TT LV1 error"},
	{ ((uint64_t)1<<45), "TT_BUNCH_X_ERROR" , "TT bunch-x error"}

      };

    return ERRORDICT[i];
  }
};
#endif
