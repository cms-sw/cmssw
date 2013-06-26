#include "Calibration/Tools/interface/calibXMLwriter.h"
#include <string>
#include <cstdio>

calibXMLwriter::calibXMLwriter(EcalSubdetector subdet) : subdet_(subdet)
{
    
    char filename[128];
    if (subdet_==EcalEndcap) {
      sprintf(filename,"endcap_iniCalib.xml");
    } else {
      sprintf(filename,"barrel_iniCalib.xml");
    }
    FILENAME = fopen(filename,"w");
    fprintf(FILENAME,"<?xml version=\"1.0\" ?>\n");
    fprintf(FILENAME,"<CalibrationConstants>\n");
    if (subdet==EcalEndcap) {
      fprintf(FILENAME,"<EcalEndcap>\n");
    } else {
      fprintf(FILENAME,"<EcalBarrel>\n");
    }

}

calibXMLwriter::~calibXMLwriter()
{
    if (subdet_==EcalEndcap) {
      fprintf(FILENAME,"<EcalEndcap>\n");
    } else {
      fprintf(FILENAME,"<EcalBarrel>\n");
    }
    fprintf(FILENAME,"</CalibrationConstants>\n");
    fclose(FILENAME);
}

void calibXMLwriter::writeLine(EBDetId const & det, float calib)
{
int eta=det.ieta();
int phi=det.iphi();
fprintf(FILENAME,"<Cell eta_index=\"%d\" phi_index=\"%d\" scale_factor=\"%f\"/>\n",eta,phi,calib);
}


void calibXMLwriter::writeLine(EEDetId const & det, float calib)
{
int x=det.ix();
int y=det.iy();
int z=det.zside();
fprintf(FILENAME,"<Cell x_index=\"%d\" y_index=\"%d\" z_index=\"%d\" scale_factor=\"%f\"/>\n",x,y,z,calib);
}

