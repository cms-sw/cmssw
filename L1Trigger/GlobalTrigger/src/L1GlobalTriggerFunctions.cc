// this class header
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerFunctions.h"

// system include files

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// methods

// factorial function

int factorial(int n) { return (n <= 0) ? 1 : (n * factorial(n - 1)); }

// hex string to a vector of 64-bit integers
bool hexStringToInt64(const std::string &hexString, std::vector<unsigned long long> &vecInt64) {
  int iVec = 0;
  size_t initialPos = 0;
  unsigned long long iValue = 0ULL;

  do {
    iValue = 0ULL;

    if (stringToNumber<unsigned long long>(iValue, hexString.substr(initialPos, 16), std::hex)) {
      LogTrace("L1GlobalTrigger") << "\n  String " << hexString.substr(initialPos, 16) << " converted to hex value 0x"
                                  << std::hex << iValue << std::dec << std::endl;

      vecInt64[iVec] = iValue;
    } else {
      LogTrace("L1GlobalTrigger") << "\nstringToNumber failed to convert string " << hexString.substr(initialPos, 16)
                                  << std::endl;

      return false;
    }

    initialPos = +16;
    iVec++;
  } while (hexString.size() >= (initialPos + 16));

  return true;
}
