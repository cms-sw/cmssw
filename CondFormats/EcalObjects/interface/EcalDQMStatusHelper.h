#ifndef CondFormats_EcalObjects_EcalDQMStatusHelper_H
#define CondFormats_EcalObjects_EcalDQMStatusHelper_H
/**
 * Author: Francesca
 * Created: 13 Jan 2009
 * 
 **/

class EcalDQMStatusHelper {
public:
  static const int CH_ID_ERROR = 0;
  static const int CH_GAIN_ZERO_ERROR = 1;
  static const int CH_GAIN_SWITCH_ERROR = 2;
  static const int TT_ID_ERROR = 3;
  static const int TT_SIZE_ERROR = 4;

  static const int PEDESTAL_LOW_GAIN_MEAN_ERROR = 5;
  static const int PEDESTAL_MIDDLE_GAIN_MEAN_ERROR = 6;
  static const int PEDESTAL_HIGH_GAIN_MEAN_ERROR = 7;
  static const int PEDESTAL_LOW_GAIN_RMS_ERROR = 8;
  static const int PEDESTAL_MIDDLE_GAIN_RMS_ERROR = 9;
  static const int PEDESTAL_HIGH_GAIN_RMS_ERROR = 10;

  static const int PEDESTAL_ONLINE_HIGH_GAIN_MEAN_ERROR = 11;
  static const int PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR = 12;

  static const int TESTPULSE_LOW_GAIN_MEAN_ERROR = 13;
  static const int TESTPULSE_MIDDLE_GAIN_MEAN_ERROR = 14;
  static const int TESTPULSE_HIGH_GAIN_MEAN_ERROR = 15;
  static const int TESTPULSE_LOW_GAIN_RMS_ERROR = 16;
  static const int TESTPULSE_MIDDLE_GAIN_RMS_ERROR = 17;
  static const int TESTPULSE_HIGH_GAIN_RMS_ERROR = 18;

  static const int LASER_MEAN_ERROR = 19;
  static const int LASER_RMS_ERROR = 20;
  static const int LASER_TIMING_MEAN_ERROR = 21;
  static const int LASER_TIMING_RMS_ERROR = 22;

  static const int LED_MEAN_ERROR = 23;
  static const int LED_RMS_ERROR = 24;
  static const int LED_TIMING_MEAN_ERROR = 25;
  static const int LED_TIMING_RMS_ERROR = 26;

  static const int STATUS_FLAG_ERROR = 27;

  static const int PHYSICS_BAD_CHANNEL_WARNING = 28;
  static const int PHYSICS_BAD_CHANNEL_ERROR = 29;
};

#endif
