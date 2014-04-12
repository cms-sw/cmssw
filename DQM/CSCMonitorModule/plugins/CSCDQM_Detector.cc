/*
 * =====================================================================================
 *
 *       Filename:  Detector.cc
 *
 *    Description:  Class Detector implementation
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

#ifdef CSC_RENDER_PLUGIN
#include "CSCDQM_Detector.h"
#else
#include "CSCDQM_Detector.h"
#endif

namespace cscdqm {

  /**
   * @brief  Constructor
   * @param  p_partition_x Number of efficiency partitions on X axis
   * @param  p_partition_y Number of efficiency partitions on Y axis
   * @return 
   */
  Detector::Detector(const unsigned int p_partitions_x, const unsigned int p_partitions_y) {
  
    partitions_x = p_partitions_x;
    partitions_y = p_partitions_y;
  
    unsigned int i = 0; 
    Address adr;
  
    adr.mask.layer = false;
    adr.mask.side = adr.mask.station = adr.mask.ring = adr.mask.chamber = adr.mask.cfeb = adr.mask.hv = true;
  
    /**  Creating real eta/phi boxes for available addresses */
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
                  
                boxes[i].adr = adr;
  
                float xboxmin = (x_min < x_max ? x_min : x_max);
                float xboxmax = (x_max > x_min ? x_max : x_min);
                float yboxmin = (y_min < y_max ? y_min : y_max);
                float yboxmax = (y_max > y_min ? y_max : y_min);
  
                boxes[i].xmin = xboxmin;
                boxes[i].xmax = xboxmax;
                boxes[i].ymin = yboxmin;
                boxes[i].ymax = yboxmax;
  
                /** Address box calculated successfully. Now lets cache its
                 * partition elements for performace. */
  
                unsigned int x1 = int(floor(xboxmin / PARTITION_STEP_X)) + int(partitions_x / 2);
                unsigned int x2 = int( ceil(xboxmax / PARTITION_STEP_X)) + int(partitions_x / 2);
                unsigned int y1 = int(floor(yboxmin / PARTITION_STEP_Y));
                unsigned int y2 = int( ceil(yboxmax / PARTITION_STEP_Y));
  
                for (unsigned int x = x1; x < x2; x++) {
                  for (unsigned int y = y1; y < y2; y++) {
                    
                    unsigned int index = PARTITION_INDEX(x, y);
                    PartitionMapIterator iter = partitions.find(index);
                    if (iter == partitions.end()) {
                      std::vector<unsigned int> v;
                      partitions.insert(std::make_pair(index, v));
                    }
                    partitions[index].push_back(i);
  
                  }
                }
  
                i++;
  
              }
            }
          }
        }
      }
  
    }
  
    /**  Cached the most frequently used areas */
    adr.mask.side = adr.mask.ring = adr.mask.chamber = adr.mask.layer = adr.mask.cfeb = adr.mask.hv = false;
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
  
  /**
   * @brief  Calculate station area in eta/phi space
   * @param  station Station number
   * @return Area that is being covered by station
   */
  const float Detector::Area(const unsigned int station) const {
    if (station > 0 && station <= N_STATIONS) {
      return station_area[station - 1];
    }
    return 0;
  }
  
  /**
   * @brief  Return global chamber index on his geometric location
   * @param  side Side (1,2)
   * @param  station Station
   * @param  ring Ring\
   * @param  chamber Chamber position
   * @return Global chamber index starting 1. If chamber is not existing - returns 0
   */
  unsigned int Detector::GlobalChamberIndex(unsigned int side, unsigned int station, unsigned int ring, unsigned int chamber) const {
    Address adr, iadr;
    adr.mask.side = adr.mask.station = adr.mask.ring = adr.mask.chamber = true;
    adr.mask.layer = adr.mask.cfeb = adr.mask.hv = false;
    adr.side = side;
    adr.station = station;
    adr.ring = ring;
    adr.chamber = chamber;
    iadr = adr;

    unsigned int i = 1;
    for (iadr.side = 1; iadr.side <= N_SIDES; iadr.side++) { 
      for (iadr.station = 1; iadr.station <= N_STATIONS; iadr.station++) {
        for (iadr.ring = 1; iadr.ring <= NumberOfRings(iadr.station); iadr.ring++) { 
          for (iadr.chamber = 1; iadr.chamber <= NumberOfChambers(iadr.station, iadr.ring); iadr.chamber++) {
            if (iadr == adr) {
              return i;
            }
            i += 1;
          }
        }
      }
    }
    return 0;
  }

  /**
   * @brief  Calculate address area in eta/phi space
   * @param  adr Address
   * @return Area that is being covered by address
   */
  const float Detector::Area(const Address& adr) const {
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
  const unsigned int Detector::NumberOfRings(const unsigned int station) const {
    if (station == 1) return 3;
    if (station == 2) return 2;
    if (station == 3) return 2;
    if (station == 4) return 2;
    return 0;
  }
  
  /**
   * @brief  Returns the number of chambers for the given station and ring
   * @param  station Station number (1...4)
   * @param  ring Ring number (1...3)
   * @return number of chambers
   */
  const unsigned int Detector::NumberOfChambers(const unsigned int station, const unsigned int ring) const {
    if(station == 1 && ring == 1) return 36;
    if(station == 1 && ring == 2) return 36;
    if(station == 1 && ring == 3) return 36;
    if(station == 2 && ring == 1) return 18;
    if(station == 2 && ring == 2) return 36;
    if(station == 3 && ring == 1) return 18;
    if(station == 3 && ring == 2) return 36;
    if(station == 4 && ring == 1) return 18;
    if(station == 4 && ring == 2) return 36;
    return 0;
  }
  
  /**
   * @brief  Returns the number of CFEBs per Chamber on given Station/Ring
   * @param  station Station number (1...4)
   * @param  ring Ring number (1...3)
   * @return Number of CFEBs per Chamber
   */
  const unsigned int Detector::NumberOfChamberCFEBs(const unsigned int station, const unsigned int ring) const {
    if(station == 1 && ring == 1) return 4;
    if(station == 1 && ring == 2) return 5;
    if(station == 1 && ring == 3) return 4;
    if(station == 2 && ring == 1) return 5;
    if(station == 2 && ring == 2) return 5;
    if(station == 3 && ring == 1) return 5;
    if(station == 3 && ring == 2) return 5;
    if(station == 4 && ring == 1) return 5;
    if(station == 4 && ring == 2) return 5;
    return 0;
  }
  
  /**
   * @brief   Returns the number of HVs per Chamber on given Station/Ring
   * @param  station Station number (1...4)
   * @param  ring Ring number (1...3)
   * @return Number of HVs per Chamber
   */
  const unsigned int Detector::NumberOfChamberHVs(const unsigned int station, const unsigned int ring) const {
    if(station == 1 && ring == 1) return 2;
    if(station == 1 && ring == 2) return 3;
    if(station == 1 && ring == 3) return 3;
    if(station == 2 && ring == 1) return 3;
    if(station == 2 && ring == 2) return 5;
    if(station == 3 && ring == 1) return 3;
    if(station == 3 && ring == 2) return 5;
    if(station == 4 && ring == 1) return 3;
    if(station == 4 && ring == 2) return 5;
    return 0;
  }
  
  /**
   * @brief  Prints address for debugging
   * @param  adr Address to print
   * @return 
   */
  void Detector::PrintAddress(const Address& adr) const {
  
    std::cout << "Side (" << std::boolalpha << adr.mask.side << ")"; 
    if (adr.mask.side) std::cout <<  " = " << adr.side;
  
    std::cout << ", Station (" << std::boolalpha << adr.mask.station << ")"; 
    if (adr.mask.station) std::cout << " = " << adr.station;
  
    std::cout << ", Ring (" << std::boolalpha << adr.mask.ring << ")"; 
    if (adr.mask.ring) std::cout << " = " << adr.ring;
  
    std::cout << ", Chamber (" << std::boolalpha << adr.mask.chamber << ")"; 
    if (adr.mask.chamber) std::cout << " = " << adr.chamber;
  
    std::cout << ", Layer (" << std::boolalpha << adr.mask.layer << ")"; 
    if (adr.mask.layer) std::cout << " = " << adr.layer;
  
    std::cout << ", CFEB (" << std::boolalpha << adr.mask.cfeb << ")"; 
    if (adr.mask.cfeb) std::cout << " = " << adr.cfeb;
  
    std::cout << ", HV (" << std::boolalpha << adr.mask.hv << ")"; 
    if (adr.mask.hv) std::cout << " = " << adr.hv;
  
    std::cout << std::endl;
  }
  
  /**
   * @brief  Address iterator by mask
   * @param  i Iterator
   * @param  adr Address to return
   * @param  mask for addresses
   * @return true if address was found and filled in, false - otherwise 
   */
  const bool Detector::NextAddress(unsigned int& i, const Address*& adr, const Address& mask) const {
    for(; i < N_ELEMENTS; i++ ) {
      if (boxes[i].adr == mask) {
          adr = &boxes[i].adr;
          i++;
          return true; 
      }
    }
    return false;
  }
  
  /**
   * @brief  Address box iterator by mask
   * @param  i Iterator
   * @param  adr AddressBox to return
   * @param  mask for addresses
   * @return true if address box was found and filled in, false - otherwise 
   */
  const bool Detector::NextAddressBox(unsigned int& i, const AddressBox*& box, const Address& mask) const {
  
    for(; i < N_ELEMENTS; i++ ) {
      if (boxes[i].adr == mask) {
          box = &boxes[i];
          i++;
          return true; 
        }
    }
    return false;
  }
  
  /**
   * @brief  Address box iterator by partition
   * @param  i Iterator
   * @param  px Partition x index
   * @param  py Partition y index
   * @param  box AddressBox to return
   * @return true if address box was found and filled in, false - otherwise 
   */
  const bool Detector::NextAddressBoxByPartition (unsigned int& i, const unsigned int px, const unsigned int py, AddressBox*& box) {
  
    unsigned int index = PARTITION_INDEX(px, py);
  
    PartitionMapIterator iter = partitions.find(index);
    if (iter != partitions.end()) {
      if (i < partitions[index].size()) {
        box = &boxes[partitions[index].at(i)];
        i++;
        return true; 
      }
    }
    return false;
  
  }
  
  const float Detector::Eta(const float r, const float z) const {
    if(r > 0.0 || z > 0.0) {
      float sin_theta = r / sqrt(r * r + z * z);
      float cos_theta = z / sqrt(r * r + z * z);
      return - log(sin_theta / (cos_theta + 1));
    }
    if(r == 0.0) return FLT_MAX;
    return 0.0;
  }
  
  
  /**
   * @brief   Transform eta coordinate to local canvas coordinate
   * @param  eta Eta coordinate
   * @return local canvas coordinate
   */
  const float Detector::EtaToX(const float eta) const {
    float x_min   = -2.5;
    float x_max   =  2.5;
    float eta_min = -2.5;
    float eta_max =  2.5;
    float a = (x_max - x_min) / (eta_max - eta_min);
    float b = (eta_max * x_min - eta_min * x_max) / (eta_max - eta_min);
    return a * eta + b;
  }
  
  /**
   * @brief   Transform phi coordinate to local canvas coordinate
   * @param  phi Phi coordinate
   * @return local canvas coordinate
   */
  const float Detector::PhiToY(const float phi) const {
    float y_min   = 0.0;
    float y_max   = 2.0 * 3.14159;
    float phi_min = 0.0;
    float phi_max = 2.0 * 3.14159;
    float a = (y_max - y_min) / (phi_max - phi_min);
    float b = (phi_max * y_min - phi_min * y_max) / (phi_max - phi_min);
    return a * phi + b;
  }
  
  /**
   * @brief  Get Z parameter (used in address eta/phi calculation)
   * @param  station Station Id
   * @param  ring Ring Id
   * @return Z value
   */
  const float Detector::Z(const int station, const int ring) const {
    float z_csc = 0;
    
    if(station == 1 && ring == 1) z_csc = (5834.5 + 6101.5) / 2.0;
    if(station == 1 && ring == 2) z_csc = (6790.0 + 7064.3) / 2.0;
    if(station == 1 && ring == 3) z_csc = 6888.0;
    if(station == 2) z_csc = (8098.0 + 8346.0) / 2.0;
    if(station == 3) z_csc = (9414.8 + 9166.8) / 2.0;
    if(station == 4) z_csc = 10630.0; // has to be corrected
    
    return z_csc;
  }
  
  /**
   * @brief  Get R min parameter (used in address eta/phi calculation)
   * @param  station Station Id
   * @param  ring Ring Id
   * @param  n_hv HV number
   * @return R min value
   */
  const float Detector::RMinHV(const int station, const int ring, const int n_hv) const {
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
  
  /**
   * @brief  Get R max parameter (used in address eta/phi calculation)
   * @param  station Station Id
   * @param  ring Ring Id
   * @param  n_hv HV number
   * @return R max value
   */
  const float Detector::RMaxHV(const int station, const int ring, const int n_hv) const {
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
  
  /**
   * @brief  Get Min phi boundary for particular CFEB
   * @param  station Station number
   * @param  ring Ring number
   * @param  chamber Chamber number
   * @param  cfeb CFEB number
   * @return Min phi CFEB boundary
   */
  const float Detector::PhiMinCFEB(const int station, const int ring, const int chamber, const int cfeb) const {
    float phi_min_cfeb;
    
    int n_cfeb = NumberOfChamberCFEBs(station, ring);
    int n_chambers = NumberOfChambers(station, ring);
    
    phi_min_cfeb = 0.0 + 2.0 * 3.14159 / ((float) (n_chambers)) * ((float) (chamber - 1) + (float) (cfeb - 1) / (float) (n_cfeb));
    
    return phi_min_cfeb;
  }
  
  /**
   * @brief  Get Max phi boundary for particular CFEB
   * @param  station Station number
   * @param  ring Ring number
   * @param  chamber Chamber number
   * @param  cfeb CFEB number
   * @return Max phi CFEB boundary
   */
  const float Detector::PhiMaxCFEB(const int station, const int ring, const int chamber, const int cfeb) const {
    float phi_max_cfeb;
    
    int n_cfeb = NumberOfChamberCFEBs(station, ring);
    int n_chambers = NumberOfChambers(station, ring);
    
    phi_max_cfeb = 0.0 + 2.0 * 3.14159 / (float) n_chambers * ((float) (chamber - 1) + (float) (cfeb) / (float) n_cfeb);
    
    return phi_max_cfeb;
  }
  
  /**
   * @brief  Construct address from string
   * @param  str_address Address in string
   * @param  adr Address to return
   * @return true if address was successfully created, false - otherwise
   */
  const bool Detector::AddressFromString(const std::string& str_address, Address& adr) const {
    
    std::vector<std::string> tokens;
    Utility::splitString(str_address, ",", tokens);
  
    if (tokens.size() != ADDR_SIZE) return false;
  
    for (unsigned int r = 0; r < ADDR_SIZE; r++) {
  
      std::string token = tokens.at(r);
      Utility::trimString(token);
      bool mask = false;
      unsigned int num  = 0;
  
      if (token.compare("*") != 0) {
        if(stringToNumber<unsigned int>(num, token, std::dec)) {
          mask = true;
        } else {
          return false;
        }
      }
  
      switch (r) {
        case 0:
          adr.mask.side = mask;
          adr.side = num;
          break;
        case 1:
          adr.mask.station = mask;
          adr.station = num;
          break;
        case 2:
          adr.mask.ring = mask;
          adr.ring = num;
          break;
        case 3:
          adr.mask.chamber = mask;
          adr.chamber = num;
          break;
        case 4:
          adr.mask.layer = mask;
          adr.layer = num;
          break;
        case 5:
          adr.mask.cfeb = mask;
          adr.cfeb = num;
          break;
        case 6:
          adr.mask.hv = mask;
          adr.hv = num;
      }
  
    } 
  
    return true;
  
  }

}
