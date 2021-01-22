#ifndef RecoLocalCalo_HcalRecAlgos_interface_HcalConstants_h
#define RecoLocalCalo_HcalRecAlgos_interface_HcalConstants_h

namespace hcal::constants {

  constexpr int maxSamples = 10;
  constexpr int maxPSshapeBin = 256;
  constexpr int nsPerBX = 25;
  constexpr float iniTimeShift = 92.5f;
  constexpr float invertnsPerBx = 0.04f;
  constexpr int shiftTS = 4;

}  // namespace hcal::constants

#endif  // RecoLocalCalo_HcalRecAlgos_interface_HcalConstants_h
