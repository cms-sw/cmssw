#ifndef EcalTPGTPMode_h
#define EcalTPGTPMode_h

#include "CondFormats/Serialization/interface/Serializable.h"

/*
Author: Davide Valsecchi
Date:  11/02/2021

*/

class EcalTPGTPMode {
public:
  EcalTPGTPMode();
  ~EcalTPGTPMode();

  bool EnableEBOddFilter;
  bool EnableEEOddFilter;
  bool EnableEBOddPeakFinder;
  bool EnableEEOddPeakFinder;
  bool DisableEBEvenPeakFinder;
  bool DisableEEEvenPeakFinder;
  uint16_t FenixEBStripOutput;
  uint16_t FenixEEStripOutput;
  uint16_t FenixEBStripInfobit2;
  uint16_t FenixEEStripInfobit2;
  uint16_t EBFenixTcpOutput;
  uint16_t EBFenixTcpInfobit1;
  uint16_t EEFenixTcpOutput;
  uint16_t EEFenixTcpInfobit1;
  // Wildcard parameters for future use
  uint16_t FenixPar15;
  uint16_t FenixPar16;
  uint16_t FenixPar17;
  uint16_t FenixPar18;

  // print parameters to stream:
  void print(std::ostream&) const;

  friend std::ostream& operator<<(std::ostream& out, const EcalTPGTPMode& params) {
    params.print(out);
    return out;
  }

  COND_SERIALIZABLE;
};

#endif
