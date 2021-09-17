#ifndef CondFormats_EcalObjects_EcalErrorDictionary_H
#define CondFormats_EcalObjects_EcalErrorDictionary_H

/**
 *  A dictionary of bitmasks for ECAL channel errors and their meaning
 *  This object is not meant to be stored in the offline DB, but the bits
 *  defined here are stored in EcalChannelStatus.
 *
 *  This class holds no dynamic data and all the methods are static.
 */
#include <iostream>
#include <vector>
#include <cstdint>

class EcalErrorDictionary {
public:
  struct errorDef_t {
    uint64_t bitmask;
    char shortDesc[64];
    char longDesc[128];
  };

  static uint64_t hasError(std::string shortDesc, uint64_t bitcode) { return getMask(shortDesc) & bitcode; }

  static uint64_t getMask(std::string shortDesc) {
    for (unsigned int i = 0; i < DICTSIZE; i++) {
      if (getDef(i).shortDesc == shortDesc) {
        return getDef(i).bitmask;
      }
    }
    return 0;
  }

  static void printErrors(uint64_t bitcode) {
    for (unsigned int i = 0; i < DICTSIZE; i++) {
      if (bitcode & getDef(i).bitmask) {
      }
    }
  }

  static void getErrors(std::vector<errorDef_t>& errorVec, uint64_t bitcode) {
    errorVec.clear();
    for (unsigned int i = 0; i < DICTSIZE; i++) {
      if (bitcode & getDef(i).bitmask) {
        errorVec.push_back(getDef(i));
      }
    }
  }

  static void getDictionary(std::vector<errorDef_t>& dict) {
    dict.clear();
    for (unsigned int i = 0; i < DICTSIZE; i++) {
      dict.push_back(getDef(i));
    }
  }

private:
  EcalErrorDictionary(){};   // Hidden to force static use
  ~EcalErrorDictionary(){};  // Hidden to force static use

  const static unsigned int DICTSIZE = 55;

  static errorDef_t getDef(unsigned int i) {
    const static errorDef_t ERRORDICT[DICTSIZE] = {

        {((uint64_t)1 << 0), "CH_ID_WARNING", "Channel id warning"},
        {((uint64_t)1 << 1), "CH_GAIN_ZERO_WARNING", "Channel gain zero warning"},
        {((uint64_t)1 << 2), "CH_GAIN_SWITCH_WARNING", "Channel gain switch warning"},
        {((uint64_t)1 << 3), "CH_ID_ERROR", "Channel id error"},
        {((uint64_t)1 << 4), "CH_GAIN_ZERO_ERROR", "Channel gain zero error"},
        {((uint64_t)1 << 5), "CH_GAIN_SWITCH_ERROR", "Channel gain switch error"},

        {((uint64_t)1 << 6), "TT_ID_WARNING", "TT id warning"},
        {((uint64_t)1 << 7), "TT_SIZE_WARNING", "TT size warning"},
        {((uint64_t)1 << 8), "TT_LV1_WARNING", "TT LV1 warning"},
        {((uint64_t)1 << 9), "TT_BUNCH_X_WARNING", "TT bunch-x warning"},
        {((uint64_t)1 << 10), "TT_ID_ERROR", "TT id error"},
        {((uint64_t)1 << 11), "TT_SIZE_ERROR", "TT size error"},
        {((uint64_t)1 << 12), "TT_LV1_ERROR", "TT LV1 error"},
        {((uint64_t)1 << 13), "TT_BUNCH_X_ERROR", "TT bunch-x error"},

        {((uint64_t)1 << 16), "PEDESTAL_LOW_GAIN_MEAN_WARNING", "Pedestal low gain mean amplitude outside range"},
        {((uint64_t)1 << 17), "PEDESTAL_MIDDLE_GAIN_MEAN_WARNING", "Pedestal middle gain mean amplitude outside range"},
        {((uint64_t)1 << 18), "PEDESTAL_HIGH_GAIN_MEAN_WARNING", "Pedestal high gain mean amplitude outside range"},
        {((uint64_t)1 << 19), "PEDESTAL_LOW_GAIN_MEAN_ERROR", "Pedestal low gain mean amplitude error"},
        {((uint64_t)1 << 20), "PEDESTAL_MIDDLE_GAIN_MEAN_ERROR", "Pedestal middle gain mean amplitude error"},
        {((uint64_t)1 << 21), "PEDESTAL_HIGH_GAIN_MEAN_ERROR", "Pedestal high gain mean amplitude error"},

        {((uint64_t)1 << 22), "PEDESTAL_LOW_GAIN_RMS_WARNING", "Pedestal low gain rms amplitude outside range"},
        {((uint64_t)1 << 23), "PEDESTAL_MIDDLE_GAIN_RMS_WARNING", "Pedestal middle gain rms amplitude outside range"},
        {((uint64_t)1 << 24), "PEDESTAL_HIGH_GAIN_RMS_WARNING", "Pedestal high gain rms amplitude outside range"},
        {((uint64_t)1 << 25), "PEDESTAL_LOW_GAIN_RMS_ERROR", "Pedestal low gain rms amplitude error"},
        {((uint64_t)1 << 26), "PEDESTAL_MIDDLE_GAIN_RMS_ERROR", "Pedestal middle gain rms amplitude error"},
        {((uint64_t)1 << 27), "PEDESTAL_HIGH_GAIN_RMS_ERROR", "Pedestal high gain rms amplitude error"},

        {((uint64_t)1 << 28),
         "PEDESTAL_ONLINE_HIGH_GAIN_MEAN_WARNING",
         "Pedestal online high gain mean amplitude outside range"},
        {((uint64_t)1 << 29),
         "PEDESTAL_ONLINE_HIGH_GAIN_RMS_WARNING",
         "Pedestal online high gain rms amplitude outside range"},
        {((uint64_t)1 << 30), "PEDESTAL_ONLINE_HIGH_GAIN_MEAN_ERROR", "Pedestal online high gain mean amplitude error"},
        {((uint64_t)1 << 31), "PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR", "Pedestal online high gain rms amplitude error"},

        {((uint64_t)1 << 32), "TESTPULSE_LOW_GAIN_MEAN_WARNING", "Testpulse low gain mean amplitude outside range"},
        {((uint64_t)1 << 33),
         "TESTPULSE_MIDDLE_GAIN_MEAN_WARNING",
         "Testpulse middle gain mean amplitude outside range"},
        {((uint64_t)1 << 34), "TESTPULSE_HIGH_GAIN_MEAN_WARNING", "Testpulse high gain mean amplitude outside range"},
        {((uint64_t)1 << 35), "TESTPULSE_LOW_GAIN_RMS_WARNING", "Testpulse low gain rms amplitude outside range"},
        {((uint64_t)1 << 36), "TESTPULSE_MIDDLE_GAIN_RMS_WARNING", "Testpulse middle gain rms amplitude outside range"},
        {((uint64_t)1 << 37), "TESTPULSE_HIGH_GAIN_RMS_WARNING", "Testpulse high gain rms amplitude outside range"},

        {((uint64_t)1 << 38), "LASER_MEAN_WARNING", "Laser mean amplitude outside range"},
        {((uint64_t)1 << 39), "LASER_RMS_WARNING", "Laser rms amplitude outside range"},

        {((uint64_t)1 << 40), "LASER_MEAN_OVER_PN_WARNING", "Laser mean amplitude over PN outside range"},
        {((uint64_t)1 << 41), "LASER_RMS_OVER_PN_WARNING", "Laser rms amplitude over PN outside range"},

        {((uint64_t)1 << 42), "LASER_MEAN_TIMING_WARNING", "Laser channel mean timing outside range"},
        {((uint64_t)1 << 43), "LASER_RMS_TIMING_WARNING", "Laser channel rms timing outside range"},

        {((uint64_t)1 << 44), "LASER_MEAN_TT_TIMING_WARNING", "Laser tower mean timing outside range"},
        {((uint64_t)1 << 45), "LASER_RMS_TT_TIMING_WARNING", "Laser tower rms timing outside range"},

        {((uint64_t)1 << 46), "PHYSICS_MEAN_TIMING_WARNING", "Channel mean timing outside range for physics events"},
        {((uint64_t)1 << 47), "PHYSICS_RMS_TIMING_WARNING", "Channel rms timing outside range for physics events"},

        {((uint64_t)1 << 48), "PHYSICS_MEAN_TT_TIMING_WARNING", "TT mean timing outside range for physics events"},
        {((uint64_t)1 << 49), "PHYSICS_RMS_TT_TIMING_WARNING", "TT rms timing outside range for physics events"},

        {((uint64_t)1 << 50), "PHYSICS_BAD_CHANNEL_WARNING", "Bad signal for physics events"},
        {((uint64_t)1 << 51), "PHYSICS_BAD_CHANNEL_ERROR", "No signal for physics events"},

        {((uint64_t)1 << 52), "STATUS_FLAG_ERROR", "Readout tower front end error (any type)"},

        {((uint64_t)1 << 53), "LED_MEAN_WARNING", "Led mean amplitude outside range"},
        {((uint64_t)1 << 54), "LED_RMS_WARNING", "Led rms amplitude outside range"}

    };

    return ERRORDICT[i];
  }
};
#endif
