#ifndef EcalDQMStatusDictionary_H
#define EcalDQMStatusDictionary_H

/*!
  \file Status.h
  \brief dictionary for Ecal DQM status codes
  \author G. Della Ricca
  \version $Revision: 1.6 $
  \date $Date: 2010/08/08 10:07:42 $
*/

#include <boost/cstdint.hpp>
#include <iostream>
#include <vector>

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

class EcalDQMStatusDictionary {

 public:

  struct codeDef {
    uint32_t code;
    char desc[39];
  };

  static void getDictionary(std::vector<codeDef> &dict) {
    dict.clear();
    for (unsigned int i=0; i<DICTSIZE+DICTSIZE2; i++) dict.push_back(getDef(i));
  }

  static void getCodes(std::vector<codeDef>& codeVec, uint32_t code) {
      codeVec.clear();
      for (unsigned int i=0; i<DICTSIZE; i++) {
	if (code & getDef(i).code) {
	  codeVec.push_back(getDef(i));
      }
    }
  }

 private:

  EcalDQMStatusDictionary() {}; // Hidden to force static use
  ~EcalDQMStatusDictionary() {};  // Hidden to force static use

  const static unsigned int DICTSIZE = 30;
  const static unsigned int DICTSIZE2 = 30;

  static codeDef getDef(unsigned int i) {
    const static codeDef DICT[DICTSIZE+DICTSIZE2] =
      {

	{ ((uint32_t) 1 << EcalDQMStatusHelper::CH_ID_ERROR), "CH_ID_ERROR" },
	{ ((uint32_t) 1 << EcalDQMStatusHelper::CH_GAIN_ZERO_ERROR), "CH_GAIN_ZERO_ERROR" },
	{ ((uint32_t) 1 << EcalDQMStatusHelper::CH_GAIN_SWITCH_ERROR), "CH_GAIN_SWITCH_ERROR" },
	{ ((uint32_t) 1 << EcalDQMStatusHelper::TT_ID_ERROR), "TT_ID_ERROR"},
	{ ((uint32_t) 1 << EcalDQMStatusHelper::TT_SIZE_ERROR), "TT_SIZE_ERROR"},

	{ ((uint32_t) 1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_MEAN_ERROR), "PEDESTAL_LOW_GAIN_MEAN_ERROR"},
	{ ((uint32_t) 1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_MEAN_ERROR), "PEDESTAL_MIDDLE_GAIN_MEAN_ERROR"},
	{ ((uint32_t) 1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_MEAN_ERROR), "PEDESTAL_HIGH_GAIN_MEAN_ERROR"},
	{ ((uint32_t) 1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_RMS_ERROR), "PEDESTAL_LOW_GAIN_RMS_ERROR"},
	{ ((uint32_t) 1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_RMS_ERROR), "PEDESTAL_MIDDLE_GAIN_RMS_ERROR"},
	{ ((uint32_t) 1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_RMS_ERROR), "PEDESTAL_HIGH_GAIN_RMS_ERROR"},

	{ ((uint32_t) 1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_MEAN_ERROR), "PEDESTAL_ONLINE_HIGH_GAIN_MEAN_ERROR"},
	{ ((uint32_t) 1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR), "PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR"},

	{ ((uint32_t) 1 << EcalDQMStatusHelper::TESTPULSE_LOW_GAIN_MEAN_ERROR), "TESTPULSE_LOW_GAIN_MEAN_ERROR"},
	{ ((uint32_t) 1 << EcalDQMStatusHelper::TESTPULSE_MIDDLE_GAIN_MEAN_ERROR), "TESTPULSE_MIDDLE_GAIN_MEAN_ERROR"},
	{ ((uint32_t) 1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_MEAN_ERROR), "TESTPULSE_HIGH_GAIN_MEAN_ERROR"},
	{ ((uint32_t) 1 << EcalDQMStatusHelper::TESTPULSE_LOW_GAIN_RMS_ERROR), "TESTPULSE_LOW_GAIN_RMS_ERROR"},
	{ ((uint32_t) 1 << EcalDQMStatusHelper::TESTPULSE_MIDDLE_GAIN_RMS_ERROR), "TESTPULSE_MIDDLE_GAIN_RMS_ERROR"},
	{ ((uint32_t) 1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_RMS_ERROR), "TESTPULSE_HIGH_GAIN_RMS_ERROR"},

	{ ((uint32_t) 1 << EcalDQMStatusHelper::LASER_MEAN_ERROR), "LASER_MEAN_ERROR"},
	{ ((uint32_t) 1 << EcalDQMStatusHelper::LASER_RMS_ERROR), "LASER_RMS_ERROR"},
	{ ((uint32_t) 1 << EcalDQMStatusHelper::LASER_TIMING_MEAN_ERROR), "LASER_TIMING_MEAN_ERROR"},
	{ ((uint32_t) 1 << EcalDQMStatusHelper::LASER_TIMING_RMS_ERROR), "LASER_TIMING_RMS_ERROR"},

	{ ((uint32_t) 1 << EcalDQMStatusHelper::LED_MEAN_ERROR), "LED_MEAN_ERROR"},
	{ ((uint32_t) 1 << EcalDQMStatusHelper::LED_RMS_ERROR), "LED_RMS_ERROR"},
	{ ((uint32_t) 1 << EcalDQMStatusHelper::LED_TIMING_MEAN_ERROR), "LED_TIMING_MEAN_ERROR"},
	{ ((uint32_t) 1 << EcalDQMStatusHelper::LED_TIMING_RMS_ERROR), "LED_TIMING_RMS_ERROR"},

	{ ((uint32_t) 1 << EcalDQMStatusHelper::STATUS_FLAG_ERROR), "STATUS_FLAG_ERROR"},

	{ ((uint32_t) 1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_WARNING), "PHYSICS_BAD_CHANNEL_WARNING"},
	{ ((uint32_t) 1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_ERROR), "PHYSICS_BAD_CHANNEL_ERROR"},

// compatibility stuff

	{ ((uint32_t) 1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_MEAN_ERROR), "PEDESTAL_LOW_GAIN_MEAN_WARNING"},
	{ ((uint32_t) 1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_MEAN_ERROR), "PEDESTAL_MIDDLE_GAIN_MEAN_WARNING"},
	{ ((uint32_t) 1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_MEAN_ERROR), "PEDESTAL_HIGH_GAIN_MEAN_WARNING"},
	{ ((uint32_t) 1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_RMS_ERROR), "PEDESTAL_LOW_GAIN_RMS_WARNING"},
	{ ((uint32_t) 1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_RMS_ERROR), "PEDESTAL_MIDDLE_GAIN_RMS_WARNING"},
	{ ((uint32_t) 1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_RMS_ERROR), "PEDESTAL_HIGH_GAIN_RMS_WARNING"},

	{ ((uint32_t) 1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_MEAN_ERROR), "PEDESTAL_ONLINE_HIGH_GAIN_MEAN_WARNING"},
	{ ((uint32_t) 1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR), "PEDESTAL_ONLINE_HIGH_GAIN_RMS_WARNING"},

        { ((uint32_t) 1 << EcalDQMStatusHelper::TESTPULSE_LOW_GAIN_MEAN_ERROR), "TESTPULSE_LOW_GAIN_MEAN_WARNING"},
        { ((uint32_t) 1 << EcalDQMStatusHelper::TESTPULSE_MIDDLE_GAIN_MEAN_ERROR), "TESTPULSE_MIDDLE_GAIN_MEAN_WARNING"},
        { ((uint32_t) 1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_MEAN_ERROR), "TESTPULSE_HIGH_GAIN_MEAN_WARNING"},
        { ((uint32_t) 1 << EcalDQMStatusHelper::TESTPULSE_LOW_GAIN_RMS_ERROR), "TESTPULSE_LOW_GAIN_RMS_WARNING"},
        { ((uint32_t) 1 << EcalDQMStatusHelper::TESTPULSE_MIDDLE_GAIN_RMS_ERROR), "TESTPULSE_MIDDLE_GAIN_RMS_WARNING"},
        { ((uint32_t) 1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_RMS_ERROR), "TESTPULSE_HIGH_GAIN_RMS_WARNING"},

        { ((uint32_t) 1 << EcalDQMStatusHelper::LASER_MEAN_ERROR), "LASER_MEAN_WARNING"},
        { ((uint32_t) 1 << EcalDQMStatusHelper::LASER_RMS_ERROR), "LASER_RMS_WARNING"},
        { ((uint32_t) 1 << EcalDQMStatusHelper::LASER_TIMING_MEAN_ERROR), "LASER_MEAN_TIMING_WARNING"},
        { ((uint32_t) 1 << EcalDQMStatusHelper::LASER_TIMING_RMS_ERROR), "LASER_RMS_TIMING_WARNING"},

	{ ((uint32_t) 1 << EcalDQMStatusHelper::LASER_MEAN_ERROR), "LASER_MEAN_OVER_PN_ERROR"},
	{ ((uint32_t) 1 << EcalDQMStatusHelper::LASER_RMS_ERROR), "LASER_RMS_OVER_PN_ERROR"},
	{ ((uint32_t) 1 << EcalDQMStatusHelper::LASER_TIMING_MEAN_ERROR), "LASER_MEAN_TT_TIMING_ERROR"},
	{ ((uint32_t) 1 << EcalDQMStatusHelper::LASER_TIMING_RMS_ERROR), "LASER_RMS_TT_TIMING_ERROR"},

        { ((uint32_t) 1 << EcalDQMStatusHelper::LASER_MEAN_ERROR), "LASER_MEAN_OVER_PN_WARNING"},
        { ((uint32_t) 1 << EcalDQMStatusHelper::LASER_RMS_ERROR), "LASER_RMS_OVER_PN_WARNING"},
        { ((uint32_t) 1 << EcalDQMStatusHelper::LASER_TIMING_MEAN_ERROR), "LASER_MEAN_TT_TIMING_WARNING"},
        { ((uint32_t) 1 << EcalDQMStatusHelper::LASER_TIMING_RMS_ERROR), "LASER_RMS_TT_TIMING_WARNING"},

        { ((uint32_t) 1 << EcalDQMStatusHelper::LED_MEAN_ERROR), "LED_MEAN_WARNING"},
        { ((uint32_t) 1 << EcalDQMStatusHelper::LED_RMS_ERROR), "LED_RMS_WARNING"},
        { ((uint32_t) 1 << EcalDQMStatusHelper::LED_TIMING_MEAN_ERROR), "LED_MEAN_TT_TIMING_WARNING"},
        { ((uint32_t) 1 << EcalDQMStatusHelper::LED_TIMING_RMS_ERROR), "LED_RMS_TT_TIMING_WARNING"}

      };

    return DICT[i];
  }


};

#endif // EcalDQMStatusDictionary_H
