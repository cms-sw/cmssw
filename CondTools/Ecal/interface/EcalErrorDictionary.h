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

  const static unsigned int DICTSIZE = 27;
  
  static errorDef_t getDef(unsigned int i) {
    const static errorDef_t ERRORDICT[DICTSIZE] =
      {

	{ ((uint64_t)1<<0) , "PEDESTAL_MEAN_AMPLITUDE_TOO_LOW" , "Pedestal mean amplitude too low"},
	{ ((uint64_t)1<<1) , "PEDESTAL_MEAN_AMPLITUDE_TOO_HIGH" , "Pedestal mean amplitude too high"},
	{ ((uint64_t)1<<2) , "PEDESTAL_RMS_AMPLITUDE_TOO_LOW" , "Pedestal RMS amplitude too low"},
	{ ((uint64_t)1<<3) , "PEDESTAL_RMS_AMPLITUDE_TOO_HIGH" , "Pedestal RMS amplitude too high"},
	{ ((uint64_t)1<<4) , "PEDESTAL_ONLINE_MEAN_AMPLITUDE_TOO_LOW" , "Pedestal online mean amplitude too low"},
	{ ((uint64_t)1<<5) , "PEDESTAL_ONLINE_MEAN_AMPLITUDE_TOO_HIGH" ,  "Pedestal online mean amplitude too high"},
	{ ((uint64_t)1<<6) , "PEDESTAL_ONLINE_RMS_AMPLITUDE_TOO_LOW" , "Pedestal online RMS amplitude too low"},
	{ ((uint64_t)1<<7) , "PEDESTAL_ONLINE_RMS_AMPLITUDE_TOO_HIGH" , "Pedestal online RMS amplitude too high"},

	{ ((uint64_t)1<<8) , "TESTPULSE_MEAN_AMPLITUDE_TOO_LOW" , "Testpulse mean amplitude too low"},
	{ ((uint64_t)1<<9) , "TESTPULSE_MEAN_AMPLITUDE_TOO_HIGH" , "Testpulse mean amplitude too high"},
	{ ((uint64_t)1<<10), "TESTPULSE_RMS_AMPLITUDE_TOO_LOW" , "Testpulse RMS mean amplitude too low"},
	{ ((uint64_t)1<<11), "TESTPULSE_RMS_AMPLITUDE_TOO_HIGH" , "Testpulse RMS mean amplitude too high"},

	{ ((uint64_t)1<<16), "LASER_MEAN_AMPLITUDE_TOO_LOW" , "Laser mean amplitude too low"},
	{ ((uint64_t)1<<17), "LASER_MEAN_AMPLITUDE_TOO_HIGH" , "Laser mean amplitude too high"},
	{ ((uint64_t)1<<18), "LASER_RMS_AMPLITUDE_TOO_LOW" , "Laser RMS amplitude too low"},
	{ ((uint64_t)1<<19), "LASER_RMS_AMPLITUDE_TOO_HIGH" , "Laser RMS amplitude too high"},

	{ ((uint64_t)1<<24), "LASER_MEAN_AMPLITUDE_OVER_PN_TOO_LOW" , "Laser mean amplitude over PN too low"},
	{ ((uint64_t)1<<25), "LASER_MEAN_AMPLITUDE_OVER_PN_TOO_HIGH" , "Laser mean amplitude over PN too high"},
	{ ((uint64_t)1<<26), "LASER_RMS_AMPLITUDE_OVER_PN_TOO_LOW" , "Laser RMS amplitude over PN too low"},
	{ ((uint64_t)1<<27), "LASER_RMS_AMPLITUDE_OVER_PN_TOO_HIGH" , "Laser RMS amplitude over PN too high"},

	{ ((uint64_t)1<<32), "CRYSTAL_CONSISTENCY_ID" , "Crystal consistency id"},
	{ ((uint64_t)1<<33), "CRYSTAL_CONSISTENCY_GAIN_ZERO" , "Crystal consistency gain zero"},
	{ ((uint64_t)1<<34), "CRYSTAL_CONSISTENCY_GAIN_SWITCH" , "Crystal consistency gain switch"},

	{ ((uint64_t)1<<36), "TT_CONSISTENCY_ID" , "TT consistency id"},
	{ ((uint64_t)1<<37), "TT_CONSISTENCY_SIZE" , "TT consistency size"},
	{ ((uint64_t)1<<38), "TT_CONSISTENCY_LV1" , "TT consistency LV1"},
	{ ((uint64_t)1<<39), "TT_CONSISTENCY_BUNCH_X" , "TT consistency bunch X"}

      };

    return ERRORDICT[i];
  }
};
#endif
