#include "Calibration/Tools/interface/Pi0CalibXMLwriter.h"
#include <string>
#include <cstdio>

Pi0CalibXMLwriter::Pi0CalibXMLwriter(EcalSubdetector subdet) : subdet_(subdet) {
  char filename[128];
  if (subdet_ == EcalEndcap) {
    sprintf(filename, "endcap_iniCalib.xml");
  } else {
    sprintf(filename, "barrel_iniCalib.xml");
  }
  FILENAME = fopen(filename, "w");
  fprintf(FILENAME, "<?xml version=\"1.0\" ?>\n");
  fprintf(FILENAME, "<CalibrationConstants>\n");
  if (subdet == EcalEndcap) {
    fprintf(FILENAME, "<EcalEndcap>\n");
  } else {
    fprintf(FILENAME, "<EcalBarrel>\n");
  }
}

Pi0CalibXMLwriter::Pi0CalibXMLwriter(EcalSubdetector subdet, int loop) : subdet_(subdet), loop_(loop) {
  char filename[128];
  if (subdet_ == EcalEndcap) {
    sprintf(filename, "endcap_calib_loop_%d.xml", loop);
  } else {
    sprintf(filename, "barrel_calib_loop_%d.xml", loop);
  }
  FILENAME = fopen(filename, "w");
  fprintf(FILENAME, "<?xml version=\"1.0\" ?>\n");
  fprintf(FILENAME, "<CalibrationConstants>\n");
  if (subdet == EcalEndcap) {
    fprintf(FILENAME, "<EcalEndcap>\n");
  } else {
    fprintf(FILENAME, "<EcalBarrel>\n");
  }
}

Pi0CalibXMLwriter::~Pi0CalibXMLwriter() {
  if (subdet_ == EcalEndcap) {
    fprintf(FILENAME, "<EcalEndcap>\n");
  } else {
    fprintf(FILENAME, "<EcalBarrel>\n");
  }
  fprintf(FILENAME, "</CalibrationConstants>\n");
  fclose(FILENAME);
}

void Pi0CalibXMLwriter::writeLine(EBDetId const& det, float calib) {
  int eta = det.ieta();
  int phi = det.iphi();
  fprintf(FILENAME, "<Cell eta_index=\"%d\" phi_index=\"%d\" scale_factor=\"%f\"/>\n", eta, phi, calib);
}

void Pi0CalibXMLwriter::writeLine(EEDetId const& det, float calib) {
  int x = det.ix();
  int y = det.iy();
  int z = det.zside() > 0 ? 1 : 0;
  fprintf(FILENAME, "<Cell x_index=\"%d\" y_index=\"%d\" z_index=\"%d\" scale_factor=\"%f\"/>\n", x, y, z, calib);
}
