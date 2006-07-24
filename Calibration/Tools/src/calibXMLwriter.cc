#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Calibration/Tools/interface/calibXMLwriter.h"
#include <iostream>
#include <string>


calibXMLwriter::calibXMLwriter()
{
    
    char filename[128];
    sprintf(filename,"barrel_iniCalib.xml");
    FILENAME = fopen(filename,"w");
    fprintf(FILENAME,"<?xml version=\"1.0\" ?>\n");
    fprintf(FILENAME,"<CalibrationConstants>\n");
    fprintf(FILENAME,"<EcalBarrel>\n");

}

calibXMLwriter::~calibXMLwriter()
{
    fprintf(FILENAME,"</EcalBarrel>\n");
    fprintf(FILENAME,"</CalibrationConstants>\n");
    fclose(FILENAME);
}

void calibXMLwriter::writeLine(EBDetId const & det, float calib)
{
int eta=det.ieta();
int phi=det.iphi();
fprintf(FILENAME,"<Cell eta_index=\"%d\" phi_index=\"%d\" scale_factor=\"%f\"/>\n",eta,phi,calib);
}

