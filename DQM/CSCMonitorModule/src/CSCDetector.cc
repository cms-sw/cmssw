/*
 * =====================================================================================
 *
 *       Filename:  CSCDetector.cc
 *
 *    Description:  Class CSCDetector implementation
 *
 *        Version:  1.0
 *        Created:  05/19/2008 10:59:34 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), Valdas.Rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#include "DQM/CSCMonitorModule/interface/CSCDetector.h"

/**
 * @brief  Constructor
 * @param  
 * @return 
 */
CSCDetector::CSCDetector() {

  unsigned int i = 0; 
  CSCAddress adr;

  float x_min_chamber = 0, x_max_chamber = 0, y_min_chamber = 0, y_max_chamber = 0;

  adr.mask.side = adr.mask.station = adr.mask.ring = adr.mask.chamber = adr.mask.cfeb = adr.mask.hv = true;

  for (adr.side = 1; adr.side <= N_SIDES; adr.side++) { 
    float sign = +1.0;
    if(adr.side == 2) sign = -1.0;
    for (adr.station = 1; adr.station <= N_STATIONS; adr.station++) {
       for (adr.ring = 1; adr.ring <= NumberOfRings(adr.station); adr.ring++) { 
          for (adr.chamber = 1; adr.chamber <= NumberOfChambers(adr.station, adr.ring); adr.chamber++) {
            for (adr.cfeb = 1; adr.cfeb <= NumberOfChamberCFEBs(adr.station, adr.ring); adr.cfeb++) {
              for (adr.hv = 1; adr.hv <= NumberOfChamberHVs(adr.station, adr.ring); adr.hv++) {

                float z = Z(adr.station, adr.ring);
                float r_min = RMinHV(adr.station, adr.ring, adr.hv);
                float r_max = RMaxHV(adr.station, adr.ring, adr.hv);
                float eta_min = sign * Eta(r_min, z);
                float eta_max = sign * Eta(r_max, z);
                float x_min = EtaToX(eta_min);
                float x_max = EtaToX(eta_max);
                float phi_min = 0;
                float phi_max = 0;
                if(adr.station == 1 && adr.ring == 1 && adr.hv == 1) {
                  phi_min = PhiMinCFEB(adr.station, adr.ring, adr.chamber, 1);
                  phi_max = PhiMaxCFEB(adr.station, adr.ring, adr.chamber, NumberOfChamberCFEBs(adr.station, adr.ring));
                } else {
                  phi_min = PhiMinCFEB(adr.station, adr.ring, adr.chamber, adr.cfeb);
                  phi_max = PhiMaxCFEB(adr.station, adr.ring, adr.chamber, adr.cfeb);
                }
                float y_min = PhiToY(phi_min);
                float y_max = PhiToY(phi_max);
                if(adr.cfeb == 1 && adr.hv == 1) {
                  x_min_chamber = x_min;
                  y_min_chamber = y_min;
                }
                if(adr.cfeb == NumberOfChamberCFEBs(adr.station, adr.ring) && adr.hv == NumberOfChamberHVs(adr.station, adr.ring)) {
                  x_max_chamber = x_max;
                  y_max_chamber = y_max;
                }
                
                boxes[i].adr = adr;
                boxes[i].xmin = x_min;
                boxes[i].xmax = x_max;
                boxes[i].ymin = y_min;
                boxes[i].ymax = y_max;
                i++;

              }
            }
          }
       }
    }
  }

  // Cached the most frequently used areas
  adr.mask.side = adr.mask.ring = adr.mask.chamber = adr.mask.cfeb = adr.mask.hv = false;
  adr.mask.station = true;
  adr.station = 1;
  station_area[0] = Area(adr);
  adr.station = 2;
  station_area[1] = Area(adr);
  adr.station = 3;
  station_area[2] = Area(adr);
  adr.station = 4;
  station_area[3] = Area(adr);
}

const float CSCDetector::Area(const unsigned int station) const {
  if (station > 0 && station <= N_STATIONS) {
    return station_area[station - 1];
  }
  return 0;
}

const float CSCDetector::Area(const CSCAddress& adr) const {
  float a = 0;
  for(unsigned int i = 0; i < N_ELEMENTS; i++ ) {
    if (boxes[i].adr == adr) {
      a += fabs((boxes[i].xmax - boxes[i].xmin) * (boxes[i].ymax - boxes[i].ymin));
    }
  }
  return a;
}

/**
 * @brief  Returns the number of rings for the given station
 * @param  station Station number (1, 2, 3, 4)
 * @return number of rings for the given station
 */
const unsigned int CSCDetector::NumberOfRings(const unsigned int station) const {
  switch (station) {
    case 1:
      return 3;
    case 2:
      return 2;
    case 3:
      return 2;
    case 4:
      return 1;
  }
  return 0;
}

/**
 * @brief  Returns the number of chambers for the given station and ring
 * @param  station Station number (1...4)
 * @param  ring Ring number (1...3)
 * @return number of chambers
 */
const unsigned int CSCDetector::NumberOfChambers(const unsigned int station, const unsigned int ring) const {
  if(station == 1 && ring == 1) return 36;
  if(station == 1 && ring == 2) return 36;
  if(station == 1 && ring == 3) return 36;
  if(station == 2 && ring == 1) return 18;
  if(station == 2 && ring == 2) return 36;
  if(station == 3 && ring == 1) return 18;
  if(station == 3 && ring == 2) return 36;
  if(station == 4 && ring == 1) return 18;
  return 0;
}

/**
 * @brief  Returns the number of CFEBs per Chamber on given Station/Ring
 * @param  station Station number (1...4)
 * @param  ring Ring number (1...3)
 * @return Number of CFEBs per Chamber
 */
const unsigned int CSCDetector::NumberOfChamberCFEBs(const unsigned int station, const unsigned int ring) const {
  if(station == 1 && ring == 1) return 4;
  if(station == 1 && ring == 2) return 5;
  if(station == 1 && ring == 3) return 4;
  if(station == 2 && ring == 1) return 5;
  if(station == 2 && ring == 2) return 5;
  if(station == 3 && ring == 1) return 5;
  if(station == 3 && ring == 2) return 5;
  if(station == 4 && ring == 1) return 5;
  return 0;
}

/**
 * @brief   Returns the number of HVs per Chamber on given Station/Ring
 * @param  station Station number (1...4)
 * @param  ring Ring number (1...3)
 * @return Number of HVs per Chamber
 */
const unsigned int CSCDetector::NumberOfChamberHVs(const unsigned int station, const unsigned int ring) const {
  if(station == 1 && ring == 1) return 2;
  if(station == 1 && ring == 2) return 3;
  if(station == 1 && ring == 3) return 3;
  if(station == 2 && ring == 1) return 3;
  if(station == 2 && ring == 2) return 5;
  if(station == 3 && ring == 1) return 3;
  if(station == 3 && ring == 2) return 5;
  if(station == 4 && ring == 1) return 3;
  return 0;
}

/**
 * @brief  Prints address for debugging
 * @param  adr Address to print
 * @return 
 */
void CSCDetector::PrintAddress(const CSCAddress& adr) const {

  std::cout << "Side (" << std::boolalpha << adr.mask.side << ")"; 
  if (adr.mask.side) std::cout <<  " = " << adr.side;

  std::cout << ", Station (" << std::boolalpha << adr.mask.station << ")"; 
  if (adr.mask.station) std::cout << " = " << adr.station;

  std::cout << ", Ring (" << std::boolalpha << adr.mask.ring << ")"; 
  if (adr.mask.ring) std::cout << " = " << adr.ring;

  std::cout << ", Chamber (" << std::boolalpha << adr.mask.chamber << ")"; 
  if (adr.mask.chamber) std::cout << " = " << adr.chamber;

  std::cout << ", CFEB (" << std::boolalpha << adr.mask.cfeb << ")"; 
  if (adr.mask.cfeb) std::cout << " = " << adr.cfeb;

  std::cout << ", HV (" << std::boolalpha << adr.mask.hv << ")"; 
  if (adr.mask.hv) std::cout << " = " << adr.hv;

  std::cout << std::endl;
}

const bool CSCDetector::NextAddress(unsigned int& i, const CSCAddress*& adr, const CSCAddress& mask) const {
  unsigned int c = 0;
  for(unsigned int p = 0; p < N_ELEMENTS; p++ ) {
    if (boxes[p].adr == mask) {
      if (c == i) {
        i++;
        adr = &boxes[p].adr;
        return true; 
      }
      c++;
    }
  }
  return false;
}

const bool CSCDetector::NextAddressBox(unsigned int& i, const CSCAddressBox*& box, const CSCAddress& mask) const {
  unsigned int c = 0;
  for(unsigned int p = 0; p < N_ELEMENTS; p++ ) {
    if (boxes[p].adr == mask) {
      if (c == i) {
        i++;
        box = &boxes[p];
        return true; 
      }
      c++;
    }
  }
  return false;
}

const float CSCDetector::Eta(const float r, const float z) const {
  if(r > 0.0 || z > 0.0) {
    float sin_theta, cos_theta;
    sin_theta = r/sqrt(r*r+z*z);
    cos_theta = z/sqrt(r*r+z*z);
    return  - log(sin_theta/(cos_theta + 1));
  }
  if(r == 0.0) return FLT_MAX;
  return 0.0;
}


// Transform eta coordinate to local canvas coordinate
const float CSCDetector::EtaToX(const float eta) const {
  float x_min   = -2.5;
  float x_max   =  2.5;
  float eta_min = -2.5;
  float eta_max =  2.5;
  float a = (x_max - x_min) / (eta_max - eta_min);
  float b = (eta_max * x_min - eta_min * x_max) / (eta_max - eta_min);
  return a * eta + b;
}

// Transform phi coordinate to local canvas coordinate
const float CSCDetector::PhiToY(const float phi) const {
  float y_min   = 0.0;
  float y_max   = 2.0 * 3.14159;
  float phi_min = 0.0;
  float phi_max = 2.0 * 3.14159;
  float a = (y_max - y_min) / (phi_max - phi_min);
  float b = (phi_max * y_min - phi_min * y_max) / (phi_max - phi_min);
  return a * phi + b;
}

const float CSCDetector::Z(const int station, const int ring) const {
  float z_csc = 0;
  
  if(station == 1 && ring == 1) z_csc = (5834.5 + 6101.5) / 2.0;
  if(station == 1 && ring == 2) z_csc = (6790.0 + 7064.3) / 2.0;
  if(station == 1 && ring == 3) z_csc = 6888.0;
  if(station == 2) z_csc = (8098.0 + 8346.0) / 2.0;
  if(station == 3) z_csc = (9414.8 + 9166.8) / 2.0;
  if(station == 4) z_csc = 10630.0; // has to be corrected
  
  return z_csc;
}

const float CSCDetector::RMinHV(const int station, const int ring, const int n_hv) const {
  float r_min_hv = 0;
  
  if(station == 1 && ring == 1) {
    if(n_hv == 1) r_min_hv = 1060.0;
    if(n_hv == 2) r_min_hv = 1500.0;
  }
  
  if(station == 1 && ring == 2) {
    if(n_hv == 1) r_min_hv = 2815.0;
    if(n_hv == 2) r_min_hv = 3368.2;
    if(n_hv == 3) r_min_hv = 4025.7;
  }
  
  if(station == 1 && ring == 3) {
    if(n_hv == 1) r_min_hv = 5120.0;
    if(n_hv == 2) r_min_hv = 5724.1;
    if(n_hv == 3) r_min_hv = 6230.2;
  }
  
  if(station == 2 && ring == 1) {
    if(n_hv == 1) r_min_hv = 1469.2;
    if(n_hv == 2) r_min_hv = 2152.3;
    if(n_hv == 3) r_min_hv = 2763.7;
  }
  
  if(station == 3 && ring == 1) {
    if(n_hv == 1) r_min_hv = 1668.9;
    if(n_hv == 2) r_min_hv = 2164.9;
    if(n_hv == 3) r_min_hv = 2763.8;
  }
  
  if(station == 4 && ring == 1) {
    if(n_hv == 1) r_min_hv = 1876.1;
    if(n_hv == 2) r_min_hv = 2365.9;
    if(n_hv == 3) r_min_hv = 2865.0;
  }
  
  if((station == 2 || station == 3 || station == 4) && ring == 2) {
    if(n_hv == 1) r_min_hv = 3640.2;
    if(n_hv == 2) r_min_hv = 4446.3;
    if(n_hv == 3) r_min_hv = 5053.2;
    if(n_hv == 4) r_min_hv = 5660.1;
    if(n_hv == 5) r_min_hv = 6267.0;
  }
  
  return r_min_hv;
}

const float CSCDetector::RMaxHV(const int station, const int ring, const int n_hv) const {
  float r_max_hv = 0;
  
  if(station == 1 && ring == 1) {
    if(n_hv == 1) r_max_hv = 1500.0;
    if(n_hv == 2) r_max_hv = 2565.0;
  }
  
  if(station == 1 && ring == 2) {
    if(n_hv == 1) r_max_hv = 3368.2;
    if(n_hv == 2) r_max_hv = 4025.7;
    if(n_hv == 3) r_max_hv = 4559.9;
  }
  
  if(station == 1 && ring == 3) {
    if(n_hv == 1) r_max_hv = 5724.1;
    if(n_hv == 2) r_max_hv = 6230.2;
    if(n_hv == 3) r_max_hv = 6761.5;
  }
  
  if(station == 2 && ring == 1) {
    if(n_hv == 1) r_max_hv = 2152.3;
    if(n_hv == 2) r_max_hv = 2763.7;
    if(n_hv == 3) r_max_hv = 3365.8;
  }
  
  if(station == 3 && ring == 1) {
    if(n_hv == 1) r_max_hv = 2164.9;
    if(n_hv == 2) r_max_hv = 2763.8;
    if(n_hv == 3) r_max_hv = 3365.8;
  }
  
  if(station == 4 && ring == 1) {
    if(n_hv == 1) r_max_hv = 2365.9;
    if(n_hv == 2) r_max_hv = 2865.0;
    if(n_hv == 3) r_max_hv = 3356.3;
  }
  
  if((station == 2 || station == 3 || station == 4) && ring == 2) {
    if(n_hv == 1) r_max_hv = 4446.3;
    if(n_hv == 2) r_max_hv = 5053.2;
    if(n_hv == 3) r_max_hv = 5660.1;
    if(n_hv == 4) r_max_hv = 6267.0;
    if(n_hv == 5) r_max_hv = 6870.8;
  }
  
  return r_max_hv;
}

const float CSCDetector::PhiMinCFEB(const int station, const int ring, const int chamber, const int cfeb) const {
  float phi_min_cfeb;
  
  int n_cfeb = NumberOfChamberCFEBs(station, ring);
  int n_chambers = NumberOfChambers(station, ring);
  
  phi_min_cfeb = 0.0 + 2.0 * 3.14159 / ((float) (n_chambers)) * ((float) (chamber - 1) + (float) (cfeb - 1) / (float) (n_cfeb));
  
  return phi_min_cfeb;
}

const float CSCDetector::PhiMaxCFEB(const int station, const int ring, const int chamber, const int cfeb) const {
  float phi_max_cfeb;
  
  int n_cfeb = NumberOfChamberCFEBs(station, ring);
  int n_chambers = NumberOfChambers(station, ring);
  
  phi_max_cfeb = 0.0 + 2.0 * 3.14159 / (float) n_chambers * ((float) (chamber - 1) + (float) (cfeb) / (float) n_cfeb);
  
  return phi_max_cfeb;
}

