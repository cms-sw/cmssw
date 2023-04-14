//Namespaces for Phase1 and Phase2
#ifndef DataFormats_EcalDigi_EcalConstants_h
#define DataFormats_EcalDigi_EcalConstants_h

#include "FWCore/Utilities/interface/HostDeviceConstant.h"

// constants used to encode difference between corrected cc time and noncorrected cc time
// into a uInt8 value to be stored in the 8 bits set aside for the jitter error ( previously unused )
namespace ecalcctiming {
  static constexpr const float clockToNS = 25.0;          // Convert clock units to ns
  static constexpr const float nonCorrectedSlope = 1.2;   // estimates nonCorreted time from corrected time
  static constexpr const float encodingOffest = 0.32;     // offsets difference in time using clock units
  static constexpr const float encodingValue = 398.4375;  // encodes time difference into 0 - 255 int range
}  // namespace ecalcctiming

// The HOST_DEVICE_CONSTANTs can not reside in the classes directly, which is
// why they are defined in a namespace and constant pointers to them are used in the classes
namespace ecalph2 {
  constexpr unsigned int NGAINS = 2;                     // Number of CATIA gains
  HOST_DEVICE_CONSTANT float gains[NGAINS] = {10., 1.};  // CATIA gain values
}  // namespace ecalph2

namespace ecalph1 {
  constexpr unsigned int NGAINS = 4;                             // Number of MGPA gains including a zero gain that
                                                                 // could be encoded in the gain id mask
  HOST_DEVICE_CONSTANT float gains[NGAINS] = {0., 12., 6., 1.};  // MGPA gain values including a zero gain
}  // namespace ecalph1

class ecalPh2 {
public:
  static constexpr double Samp_Period = 6.25;               // ADC sampling period in ns
  static constexpr unsigned int NGAINS = ecalph2::NGAINS;   // Number of CATIA gains
  static constexpr const float *gains = ecalph2::gains;     // CATIA gain values
  static constexpr unsigned int gainId1 = 1;                // Position of gain 1 in gains array
  static constexpr unsigned int gainId10 = 0;               // Position of gain 10 in gains array
  static constexpr unsigned int sampleSize = 16;            // Number of samples per event
  static constexpr unsigned int NBITS = 12;                 // Number of available bits
  static constexpr unsigned int MAXADC = (1 << NBITS) - 1;  // 2^NBITS - 1,  ADC max range
  static constexpr unsigned int kEBChannels = 61200;        // Number of channels in the barrel
  static constexpr double maxEneEB = 2000.;                 // Max attainable energy in the barrel in GeV
                                                            // ~(MAXADC * 10(gain) * 0.05 GeV(LSB at gain 10))
  static constexpr unsigned int kNOffsets = 2000;           // Number of time offsets generated for APD pulse shape
                                                            // simulation and reused for every kNOffsets^th channel
  static constexpr unsigned int kAdcMask = 0xFFF;           // ADC sample mask for unpacking
  static constexpr unsigned int kGainIdMask = 0x1;          // Gain id mask for unpacking
};

class ecalPh1 {
public:
  static constexpr double Samp_Period = 25.;               // ADC sampling period in ns
  static constexpr unsigned int NGAINS = ecalph1::NGAINS;  // Number of MGPA gains including a zero gain
  static constexpr const float *gains = ecalph1::gains;    // MGPA gain values including a zero gain
  static constexpr unsigned int sampleSize = 10;           // Number of samples per event
  static constexpr unsigned int NBITS = 12;                // Number of available bits
  static constexpr unsigned int kNOffsets = 2000;          // Number of time offsets generated for APD pulse shape
                                                           // simulation and reused for every kNOffsets^th channel
  static constexpr unsigned int kAdcMask = 0xFFF;          // ADC sample mask for unpacking
  static constexpr unsigned int kGainIdMask = 0x3;         // Gain id mask for unpacking
};
#endif
