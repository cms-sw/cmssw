#include "CondFormats/EcalObjects/interface/EcalSRSettings.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <sstream>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <algorithm>

using namespace std;

EcalSRSettings::EcalSRSettings()
    : ebDccAdcToGeV_(0.), eeDccAdcToGeV_(0.), bxGlobalOffset_(0), automaticMasks_(0), automaticSrpSelect_(0) {}

#define SR_PRINT(a) o << #a ": " << val.a##_ << "\n";
#define SR_VPRINT(a)                              \
  o << #a;                                        \
  if (val.a##_.empty())                           \
    o << "[0.." << (val.a##_.size() - 1) << "]:"; \
  else                                            \
    o << "[]: <empty>";                           \
  for (size_t i = 0; i < val.a##_.size(); ++i)    \
    o << "\t" << val.a##_[i];                     \
  o << "\n";
#define SR_VVPRINT(a)                                  \
  if (val.a##_.empty())                                \
    o << #a "[][]: <empty>\n";                         \
  for (size_t i = 0; i < val.a##_.size(); ++i) {       \
    o << #a "[" << i << "]";                           \
    if (val.a##_.empty())                              \
      o << "[0.." << (val.a##_[i].size() - 1) << "]:"; \
    else                                               \
      o << "[]: <empty>";                              \
    for (size_t j = 0; j < val.a##_[i].size(); ++j)    \
      o << "\t" << val.a##_[i][j];                     \
    o << "\n";                                         \
  }

std::ostream& operator<<(std::ostream& o, const EcalSRSettings& val) {
  o << "# Neighbour eta range, neighborhood: (2*deltaEta+1)*(2*deltaPhi+1)\n"
       "# In the vector contains:\n"
       "#   - 1 element, then value applies to whole ECAL\n"
       "#   - 2 elements, then element 0 applies to EB, element 1 to EE\n"
       "#   - 12 elements, then element i applied to SRP (i+1)\n"
       "# SRP emulation (see SimCalorimetry/EcalSelectiveReadoutProcuders) supports\n"
       "# only 1 element mode.\n";
  SR_VPRINT(deltaEta);

  o << "\n# Neighbouring eta range, neighborhood: (2*deltaEta+1)*(2*deltaPhi+1)\n"
       "# If the vector contains...\n"
       "#   ... 1 element, then value applies to whole ECAL\n"
       "#   ... 2 elements, then element 0 applies to EB, element 1 to EE\n"
       "#   ... 12 elements, then element i applied to SRP (i+1)\n"
       "# If the vector contains...\n"
       "#   ... 1 element, then value applies to whole ECAL\n"
       "#   ... 2 elements, then element 0 applies to EB, element 1 to EE\n"
       "#   ... 12 elements, then element i applied to SRP (i+1)\n"
       "# SRP emulation (see SimCalorimetry/EcalSelectiveReadoutProcuders) supports\n"
       "# only the single-element mode.\n";
  SR_VPRINT(deltaPhi);

  o << "\n# Index of time sample (staring from 1) the first DCC weights is implied\n"
       "# If the vector contains:\n"
       "#   ... 1 element, then value applies to whole ECAL\n"
       "#   ... 2 elements, then element 0 applies to EB, element 1 to EE\n"
       "#   ... 54 elements, then element i applied to DCC (i+1) (FED ID 651+i)\n"
       "# SRP emulation (see SimCalorimetry/EcalSelectiveReadoutProcuders) supports\n"
       "# only the single-element mode.\n";
  SR_VPRINT(ecalDccZs1stSample);

  o << "\n# ADC to GeV conversion factor used in ZS filter for EB\n";
  SR_PRINT(ebDccAdcToGeV);

  o << "\n# ADC to GeV conversion factor used in ZS filter for EE\n";
  SR_PRINT(eeDccAdcToGeV);

  o << "\n# DCC ZS FIR weights: weights are rounded in such way that in Hw\n"
       "# representation (weigth*1024 rounded to nearest integer) the sum is null:\n"
       "# Each element is a vector of 6 values, the 6 weights\n"
       "# If the vector contains...\n"
       "#   ... 1 element, then the weight set applies to whole ECAL\n"
       "#   ... 2 elements, then element 0 applies to EB, element 1 to EE\n"
       "#   ... 54 elements, then element i applied to DCC (i+1) (FED ID 651+i)\n";
  SR_VVPRINT(dccNormalizedWeights);

  o << "\n# Switch to use a symetric zero suppression (cut on absolute value). For\n"
       "# studies only, for time being it is not supported by the hardware.\n"
       "# having troubles for vector<bool> with coral (3.8.0pre1), using vector<int> instead,\n"
       "# 0 means false, a value different than 0 means true.\n"
       "# If the vector contains...\n"
       "#   ... 1 element, then the weight set applies to whole ECAL\n"
       "#   ... 2 elements, then element 0 applies to EB, element 1 to EE\n"
       "#   ... 54 elements, then element i applied to DCC (i+1) (FED ID 651+i)\n"
       "#   ... 75848 elements, then:\n"
       "#          for i < 61200, element i applies to EB crystal with denseIndex i\n"
       "#                         (see EBDetId::denseIndex())\n"
       "#          for i >= 61200, element i applies to EE crystal with denseIndex (i+61200)\n"
       "#                         (see EBDetId::denseIndex())\n"
       "# SRP emulation supports only 1 element mode. Hardware does not support\n"
       "# the symetric ZS, so symetricZS = 0 for real data.\n";
  SR_VPRINT(symetricZS);

  o << "\n# ZS energy threshold in GeV to apply to low interest channels of barrel\n"
       "# If the vector contains...\n"
       "#   ... 1 element, then the weight set applies to whole ECAL\n"
       "#   ... 2 elements, then element 0 applies to EB, element 1 to EE\n"
       "#   ... 54 elements, then element i applied to DCC (i+1) (FED ID 651+i)\n"
       "# SRP emulation supports only the 2-element mode.\n"
       "# Corresponds to srpBarrelLowInterestChannelZS and srpEndcapLowInterestChannelZS\n"
       "# of python configuration file parameters\n";
  SR_VPRINT(srpLowInterestChannelZS);

  o << "\n# ZS energy threshold in GeV to apply to high interest channels of endcap\n"
       "# If the vector contains...\n"
       "#   ... 1 element, then the weight set applies to whole ECAL\n"
       "#   ... 2 elements, then element 0 applies to EB, element 1 to EE\n"
       "#   ... 54 elements, then element i applied to DCC (i+1) (FED ID 651+i)\n"
       "# SRP emulation supports only the 2-element mode.\n"
       "# Corresponds to srpBarrelLowInterestChannelZS and srpEndcapLowInterestChannelZS\n"
       "# of python configuration file parameters\n";
  SR_VPRINT(srpHighInterestChannelZS);

  //  o << "\n# Switch to run w/o trigger primitive. For debug use only\n"
  //  "# having troubles for vector<bool> with coral (3.8.0pre1), using vector<int> instead\n"
  //  "# Parameter only relevant for emulation. For real data, must be contains 1 element with\n"
  //  "# value 0.\n"
  //  "#   ... 1 element, then the weight set applies to whole ECAL\n"
  //  "#   ... 2 elements, then element 0 applies to EB, element 1 to EE\n"
  //  "#   ... 54 elements, then element i applied to DCC (i+1) (FED ID 651+i)\n"
  //  "# SRP emulation supports only the single-element mode.\n";
  //  SR_VPRINT(trigPrimBypass);\n"
  //
  //  o << "\n# Mode selection for "# Trig bypass" mode\n"
  //  "# 0: TT thresholds applied on sum of crystal Et's\n"
  //  "# 1: TT thresholds applies on compressed Et from Trigger primitive\n"
  //  "# @see trigPrimByPass switch\n"
  //  "# Parameter only relevant for \n";
  //  SR_VPRINT(trigPrimBypassMode);
  //
  //  o << "\n# for debug mode only:\n";
  //  SR_VPRINT( trigPrimBypassLTH);
  //
  //  o << "\n# for debug mode only:\n";
  //  SR_VPRINT(trigPrimBypassHTH);
  //
  //  o << "\n# for debug mode only\n"
  //  "# having troubles for vector<bool> with coral (3.8.0pre1), using vector<int> instead\n";
  //  SR_VPRINT( trigPrimBypassWithPeakFinder);
  //
  //  o << "\n# Trigger Tower Flag to use when a flag is not found from the input\n"
  //  "# Trigger Primitive collection. Must be one of the following values:\n"
  //  "# 0: low interest, 1: mid interest, 3: high interest\n"
  //  "# 4: forced low interest, 5: forced mid interest, 7: forced high interest\n";
  //  SR_VPRINT(defaultTtf);\n"

  o << "\n# SR->action flag map. 4 elements\n"
       "# action_[i]: action for flag value i\n";
  SR_VPRINT(actions);

  o << "\n# Masks for TTC inputs of SRP cards\n"
       "# One element per TCC, that is 108 elements: element i applies to TCC (i+1)\n";
  SR_VPRINT(tccMasksFromConfig);

  o << "\n# Masks for SRP-SRP inputs of SRP cards\n"
       "# One element per SRP, that is 12 elements: element i applies to SRP (i+1)\n"
       "# indices: [iSrp][iCh]\n";
  SR_VVPRINT(srpMasksFromConfig);

  o << "\n# Masks for DCC output of SRP cards\n"
       "# One element per DCC, that is 54 elements: element i applies to DCC (i+1)\n";
  SR_VPRINT(dccMasks);

  o << "\n# Mask to enable pattern test. Typical value: 0.\n"
       "# One element per SRP, that is 12 elements: element i applies to SRP (i+1)\n";
  SR_VPRINT(srfMasks);

  o << "\n# Substitution flags used in patterm mode\n"
       "# indices: [iSrp][iFlag]\n";
  SR_VVPRINT(substitutionSrfs);

  o << "\n# Tester mode configuration\n";
  SR_VPRINT(testerTccEmuSrpIds);
  SR_VPRINT(testerSrpEmuSrpIds);
  SR_VPRINT(testerDccTestSrpIds);
  SR_VPRINT(testerSrpTestSrpIds);
  //@}

  o << "\n# Per SRP card bunch crossing counter offset.\n"
       "# This offset is added to the bxGlobalOffset\n";
  SR_VPRINT(bxOffsets);

  o << "\n# SRP system bunch crossing counter offset.\n"
       "# For each card the bxOffset[i]\n"
       "# is added to this one.\n";
  SR_PRINT(bxGlobalOffset);

  o << "\n# Switch for automatic channel masking. 0: disabled; 1: enabled. Standard  configuration: 1.\n"
       "# When enabled, if a FED is excluded from the run, the corresponding TCC inputs is automatically\n"
       "# masked (overwrites the tccInputMasks).\n";
  SR_PRINT(automaticMasks);

  o << "\n# Switch for automatic SRP card selection. 0: disabled; 1 : enabled..\n"
       "# When enabled, if all the FEDs corresponding to a given SRP is excluded from the run,\n"
       "# Then the corresponding SRP card is automatically excluded.\n";
  SR_PRINT(automaticSrpSelect);

  return o;
}
