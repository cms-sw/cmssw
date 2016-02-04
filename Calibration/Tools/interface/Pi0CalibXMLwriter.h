#ifndef PI0CALIB_XML_WRITER
#define PI0CALIB_XML_WRITER

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include <iostream>

class Pi0CalibXMLwriter {


public:

Pi0CalibXMLwriter(EcalSubdetector=EcalBarrel);
Pi0CalibXMLwriter(EcalSubdetector=EcalBarrel, int=0);
~Pi0CalibXMLwriter();

void writeLine(EBDetId const &, float);
void writeLine(EEDetId const &, float);


private:
 EcalSubdetector subdet_;
 int loop_;
 FILE* FILENAME;
};

#endif
