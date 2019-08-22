#ifndef CALIB_XML_WRITER
#define CALIB_XML_WRITER

//
// Writes out constants in xml file
// readable by EventSetup
// Author:  Lorenzo AGOSTINO

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include <iostream>

class calibXMLwriter {
public:
  calibXMLwriter(EcalSubdetector = EcalBarrel);
  ~calibXMLwriter();

  void writeLine(EBDetId const &, float);
  void writeLine(EEDetId const &, float);

private:
  EcalSubdetector subdet_;
  FILE *FILENAME;
};

#endif
