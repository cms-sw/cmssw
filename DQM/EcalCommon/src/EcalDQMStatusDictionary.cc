#include "DQM/EcalCommon/interface/EcalDQMStatusDictionary.h"

std::map<std::string, uint32_t> EcalDQMStatusDictionary::codeMap;

void
EcalDQMStatusDictionary::init()
{
  if(codeMap.size() > 0) return;

  codeMap["CH_ID_ERROR"] = 0x1 << EcalDQMStatusHelper::CH_ID_ERROR;
  codeMap["CH_GAIN_ZERO_ERROR"] = 0x1 << EcalDQMStatusHelper::CH_GAIN_ZERO_ERROR;
  codeMap["CH_GAIN_SWITCH_ERROR"] = 0x1 << EcalDQMStatusHelper::CH_GAIN_SWITCH_ERROR;
  codeMap["TT_ID_ERROR"] = 0x1 << EcalDQMStatusHelper::TT_ID_ERROR;
  codeMap["TT_SIZE_ERROR"] = 0x1 << EcalDQMStatusHelper::TT_SIZE_ERROR;

  codeMap["PEDESTAL_LOW_GAIN_MEAN_ERROR"] = 0x1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_MEAN_ERROR;
  codeMap["PEDESTAL_MIDDLE_GAIN_MEAN_ERROR"] = 0x1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_MEAN_ERROR;
  codeMap["PEDESTAL_HIGH_GAIN_MEAN_ERROR"] = 0x1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_MEAN_ERROR;
  codeMap["PEDESTAL_LOW_GAIN_RMS_ERROR"] = 0x1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_RMS_ERROR;
  codeMap["PEDESTAL_MIDDLE_GAIN_RMS_ERROR"] = 0x1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_RMS_ERROR;
  codeMap["PEDESTAL_HIGH_GAIN_RMS_ERROR"] = 0x1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_RMS_ERROR;

  codeMap["PEDESTAL_ONLINE_HIGH_GAIN_MEAN_ERROR"] = 0x1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_MEAN_ERROR;
  codeMap["PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR"] = 0x1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR;

  codeMap["TESTPULSE_LOW_GAIN_MEAN_ERROR"] = 0x1 << EcalDQMStatusHelper::TESTPULSE_LOW_GAIN_MEAN_ERROR;
  codeMap["TESTPULSE_MIDDLE_GAIN_MEAN_ERROR"] = 0x1 << EcalDQMStatusHelper::TESTPULSE_MIDDLE_GAIN_MEAN_ERROR;
  codeMap["TESTPULSE_HIGH_GAIN_MEAN_ERROR"] = 0x1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_MEAN_ERROR;
  codeMap["TESTPULSE_LOW_GAIN_RMS_ERROR"] = 0x1 << EcalDQMStatusHelper::TESTPULSE_LOW_GAIN_RMS_ERROR;
  codeMap["TESTPULSE_MIDDLE_GAIN_RMS_ERROR"] = 0x1 << EcalDQMStatusHelper::TESTPULSE_MIDDLE_GAIN_RMS_ERROR;
  codeMap["TESTPULSE_HIGH_GAIN_RMS_ERROR"] = 0x1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_RMS_ERROR;

  codeMap["LASER_MEAN_ERROR"] = 0x1 << EcalDQMStatusHelper::LASER_MEAN_ERROR;
  codeMap["LASER_RMS_ERROR"] = 0x1 << EcalDQMStatusHelper::LASER_RMS_ERROR;
  codeMap["LASER_TIMING_MEAN_ERROR"] = 0x1 << EcalDQMStatusHelper::LASER_TIMING_MEAN_ERROR;
  codeMap["LASER_TIMING_RMS_ERROR"] = 0x1 << EcalDQMStatusHelper::LASER_TIMING_RMS_ERROR;

  codeMap["LED_MEAN_ERROR"] = 0x1 << EcalDQMStatusHelper::LED_MEAN_ERROR;
  codeMap["LED_RMS_ERROR"] = 0x1 << EcalDQMStatusHelper::LED_RMS_ERROR;
  codeMap["LED_TIMING_MEAN_ERROR"] = 0x1 << EcalDQMStatusHelper::LED_TIMING_MEAN_ERROR;
  codeMap["LED_TIMING_RMS_ERROR"] = 0x1 << EcalDQMStatusHelper::LED_TIMING_RMS_ERROR;

  codeMap["STATUS_FLAG_ERROR"] = 0x1 << EcalDQMStatusHelper::STATUS_FLAG_ERROR;

  codeMap["PHYSICS_BAD_CHANNEL_WARNING"] = 0x1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_WARNING;
  codeMap["PHYSICS_BAD_CHANNEL_ERROR"] = 0x1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_ERROR;

  codeMap["disabled_channel"] =
    0x1 << EcalDQMStatusHelper::TT_SIZE_ERROR |
    0x1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_MEAN_ERROR |
    0x1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_MEAN_ERROR |
    0x1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_MEAN_ERROR |
    0x1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_MEAN_ERROR |
    0x1 << EcalDQMStatusHelper::TESTPULSE_LOW_GAIN_MEAN_ERROR |
    0x1 << EcalDQMStatusHelper::TESTPULSE_MIDDLE_GAIN_MEAN_ERROR |
    0x1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_MEAN_ERROR |
    0x1 << EcalDQMStatusHelper::LASER_MEAN_ERROR |
    0x1 << EcalDQMStatusHelper::LED_MEAN_ERROR |
    0x1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_WARNING |
    0x1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_ERROR;

  codeMap["dead_channel"] =
    0x1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_MEAN_ERROR | 
    0x1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_RMS_ERROR |
    0x1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_MEAN_ERROR |
    0x1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_RMS_ERROR |
    0x1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_MEAN_ERROR |
    0x1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_RMS_ERROR |
    0x1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_MEAN_ERROR |
    0x1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR |
    0x1 << EcalDQMStatusHelper::TESTPULSE_LOW_GAIN_MEAN_ERROR |
    0x1 << EcalDQMStatusHelper::TESTPULSE_LOW_GAIN_RMS_ERROR |
    0x1 << EcalDQMStatusHelper::TESTPULSE_MIDDLE_GAIN_MEAN_ERROR |
    0x1 << EcalDQMStatusHelper::TESTPULSE_MIDDLE_GAIN_RMS_ERROR |
    0x1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_MEAN_ERROR |
    0x1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_RMS_ERROR |
    0x1 << EcalDQMStatusHelper::LASER_MEAN_ERROR |
    0x1 << EcalDQMStatusHelper::LASER_RMS_ERROR |
    0x1 << EcalDQMStatusHelper::LASER_TIMING_MEAN_ERROR |
    0x1 << EcalDQMStatusHelper::LASER_TIMING_RMS_ERROR |
    0x1 << EcalDQMStatusHelper::LED_MEAN_ERROR |
    0x1 << EcalDQMStatusHelper::LED_RMS_ERROR |
    0x1 << EcalDQMStatusHelper::LED_TIMING_MEAN_ERROR | 
    0x1 << EcalDQMStatusHelper::LED_TIMING_RMS_ERROR |
    0x1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_WARNING |
    0x1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_ERROR;

  codeMap["pedestal_problem"] =
    0x1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_MEAN_ERROR |
    0x1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_RMS_ERROR |
    0x1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_MEAN_ERROR |
    0x1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_RMS_ERROR |
    0x1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_MEAN_ERROR |
    0x1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_RMS_ERROR |
    0x1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_MEAN_ERROR |
    0x1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR;

  codeMap["testpulse_problem"] =
    0x1 << EcalDQMStatusHelper::TESTPULSE_LOW_GAIN_MEAN_ERROR |
    0x1 << EcalDQMStatusHelper::TESTPULSE_LOW_GAIN_RMS_ERROR |
    0x1 << EcalDQMStatusHelper::TESTPULSE_MIDDLE_GAIN_MEAN_ERROR |
    0x1 << EcalDQMStatusHelper::TESTPULSE_MIDDLE_GAIN_RMS_ERROR |
    0x1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_MEAN_ERROR |
    0x1 << EcalDQMStatusHelper::TESTPULSE_HIGH_GAIN_RMS_ERROR;

  codeMap["laser_problem"] =
    0x1 << EcalDQMStatusHelper::LASER_MEAN_ERROR |
    0x1 << EcalDQMStatusHelper::LASER_RMS_ERROR |
    0x1 << EcalDQMStatusHelper::LASER_TIMING_MEAN_ERROR |
    0x1 << EcalDQMStatusHelper::LASER_TIMING_RMS_ERROR;

  codeMap["led_problem"] =
    0x1 << EcalDQMStatusHelper::LED_MEAN_ERROR |
    0x1 << EcalDQMStatusHelper::LED_RMS_ERROR |
    0x1 << EcalDQMStatusHelper::LED_TIMING_MEAN_ERROR |
    0x1 << EcalDQMStatusHelper::LED_TIMING_RMS_ERROR;

  codeMap["noise_problem"] =
    0x1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_RMS_ERROR |
    0x1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_RMS_ERROR |
    0x1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_RMS_ERROR |
    0x1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR |
    0x1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_ERROR;
}

void
EcalDQMStatusDictionary::clear()
{
  codeMap.clear();
}

uint32_t
EcalDQMStatusDictionary::getCode(std::string const& _name)
{
  std::map<std::string, uint32_t>::const_iterator itr(codeMap.find(_name));
  if(itr == codeMap.end()) return 0;
  else return itr->second;
}

std::vector<std::string>
EcalDQMStatusDictionary::getNames(uint32_t _code)
{
  std::vector<std::string> names;

  std::map<std::string, uint32_t>::const_iterator end(codeMap.end());
  for(std::map<std::string, uint32_t>::const_iterator itr(codeMap.begin()); itr != end; ++itr)
    if((itr->second & _code) != 0) names.push_back(itr->first);

  return names;
}
