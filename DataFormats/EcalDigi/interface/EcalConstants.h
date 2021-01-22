//Namespaces for Phase1 and Phase2
#ifndef CondFormats_EcalObject_EcalConstants_h
#define CondFormats_EcalObject_EcalConstants_h

class ecalPh2 {
public:
  static constexpr double Samp_Period = 6.25;
  static constexpr unsigned int NGAINS = 2;
  static constexpr float gains[NGAINS] = {10., 1.};
  static constexpr unsigned int gainId1 = 1;
  static constexpr unsigned int gainId10 = 0;
  static constexpr unsigned int sampleSize = 16;
  static constexpr unsigned int NBITS = 12;                 // number of available bits
  static constexpr unsigned int MAXADC = (1 << NBITS) - 1;  // 2^12 -1,  adc max range
  static constexpr unsigned int kEBChannels = 61200;
  static constexpr double maxEneEB = 2000.;
  static constexpr unsigned int kNOffsets = 2000;
  static constexpr unsigned int kAdcMask = 0xFFF;
  static constexpr unsigned int kGainIdMask = 0x3;

};  // namespace ecalPh2

class ecalPh1 {
public:
  static constexpr double Samp_Period = 25.;
  static constexpr unsigned int NGAINS = 4;
  static constexpr float gains[NGAINS] = {0., 12., 6., 1.};
  static constexpr unsigned int sampleSize = 10;
  static constexpr unsigned int kNOffsets = 2000;
};  // namespace ecalPh1
#endif
