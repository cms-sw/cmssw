#ifndef CALIB_XML_WRITER
#define CALIB_XML_WRITER

//
// Writes out constants in xml file   
// readable by EventSetup
// Author:  Lorenzo AGOSTINO

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include <iostream>

class calibXMLwriter {


public:

calibXMLwriter();
~calibXMLwriter();

void writeLine(EBDetId const &, float);


private:
FILE* FILENAME;
};

#endif
