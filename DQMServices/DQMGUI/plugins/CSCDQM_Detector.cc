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

#define CSC_RENDER_PLUGIN

#ifdef CSC_RENDER_PLUGIN
#include "CSCDQM_Detector.h"
#else
#include "DQM/CSCMonitorModule/interface/CSCDQM_Detector.h"
#endif

namespace cscdqm {

  /**
   * @brief  Constructor
   * @param  p_partition_x Number of efficiency partitions on X axis
   * @param  p_partition_y Number of efficiency partitions on Y axis
   * @return
   */
  Detector::Detector(unsigned int p_partitions_x, unsigned int p_partitions_y) {

    partitions_x = p_partitions_x;
    partitions_y = p_partitions_y;

    unsigned int i = 0;
    Address adr;

    adr.mask.layer = false;
    adr.mask.side = adr.mask.station = adr.mask.ring = adr.mask.chamber = adr.mask.cfeb = adr.mask.hv = true;

    /**  Creating real eta/phi boxes for available addresses */
    for (adr.side = 1; adr.side <= N_SIDES; adr.side++) {
      float sign = (float) SideSign(adr.side);
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
  float Detector::Area(unsigned int station) const {
    if (station > 0 && station <= N_STATIONS) {
      return station_area[station - 1];
    }
    return 0;
  }

  /**
   * @brief  Calculate address area in eta/phi space
   * @param  adr Address
   * @return Area that is being covered by address
   */
  float Detector::Area(const Address& adr) const {
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
  unsigned int Detector::NumberOfRings(unsigned int station) const {
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
  unsigned int Detector::NumberOfChambers(unsigned int station, unsigned int ring) const {
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
  unsigned int Detector::NumberOfChamberCFEBs(unsigned int station, unsigned int ring) const {
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
  unsigned int Detector::NumberOfChamberHVs(unsigned int station, unsigned int ring) const {
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
  bool Detector::NextAddress(unsigned int& i, const Address*& adr, const Address& mask) const {
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
  bool Detector::NextAddressBox(unsigned int& i, const AddressBox*& box, const Address& mask) const {

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
  bool Detector::NextAddressBoxByPartition (unsigned int& i, unsigned int px, unsigned int py, AddressBox*& box) {

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

  float Detector::Eta(float r, float z) const {
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
  float Detector::EtaToX(float eta) const {
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
  float Detector::PhiToY(float phi) const {
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
  float Detector::Z(const int station, const int ring) const {
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
  float Detector::RMinHV(const int station, const int ring, const int n_hv) const {
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
  float Detector::RMaxHV(const int station, const int ring, const int n_hv) const {
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
  float Detector::PhiMinCFEB(const int station, const int ring, const int chamber, const int cfeb) const {
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
  float Detector::PhiMaxCFEB(const int station, const int ring, const int chamber, const int cfeb) const {
    float phi_max_cfeb;

    int n_cfeb = NumberOfChamberCFEBs(station, ring);
    int n_chambers = NumberOfChambers(station, ring);

    phi_max_cfeb = 0.0 + 2.0 * 3.14159 / (float) n_chambers * ((float) (chamber - 1) + (float) (cfeb) / (float) n_cfeb);

    return phi_max_cfeb;
  }

  /**
   * @brief  Get the full name of the address prefixed with CSC_. It is being used by summaryReportContent variables
   * @param  adr Address
   * @return Address name as std::string
   */
  const std::string Detector::AddressName(const Address& adr) const {
    std::ostringstream oss;
    oss << "CSC";
    if (adr.mask.side) {
      oss << "_Side" << (adr.side == 1 ? "Plus" : "Minus");
      if (adr.mask.station) {
        oss << "_Station" << std::setfill('0') << std::setw(2) << adr.station;
        if (adr.mask.ring) {
          oss << "_Ring" << std::setfill('0') << std::setw(2) << adr.ring;
          if (adr.mask.chamber) {
            oss << "_Chamber" << std::setfill('0') << std::setw(2) << adr.chamber;
            if (adr.mask.layer) {
              oss << "_Layer" << std::setfill('0') << std::setw(2) << adr.layer;
              if (adr.mask.cfeb) {
                oss << "_CFEB" << std::setfill('0') << std::setw(2) << adr.cfeb;
                if (adr.mask.hv) {
                  oss << "_HV" << std::setfill('0') << std::setw(2) << adr.hv;
                }
              }
            }
          }
        }
      }
    }
    return oss.str();
  }

  /**
   * @brief  Construct address from std::string
   * @param  str_address Address in std::string
   * @param  adr Address to return
   * @return true if address was successfully created, false - otherwise
   */
  bool Detector::AddressFromString(const std::string str_address, Address& adr) const {

    std::vector<std::string> tokens;
    Utility::tokenize(str_address, tokens, ",");

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

  int Detector::NumberOfChamberParts(int station, int ring) const {
    if (station == 1 && ring == 1) return 2;
    return 1;
  }

  std::string Detector::ChamberPart(int npart) const {
    if (npart == 2) return "a";
    return "b";
  }

  bool Detector::isChamberInstalled(int side, int station, int ring, int chamber) const {
    if (station == 4 && ring == 2) {
      if (side == -1 || side == 2) return false;
      if (side ==  1 && (chamber < 9 || chamber > 13)) return false;
    }
    return true;
  }

  int Detector::SideSign(int side) const {
    return (side == 2 ? -1 : 1);
  }

  int Detector::NumberOfWiregroups(int station, int ring) const {
    if (station == 1 && ring == 1) return 48;
    if (station == 1 && ring == 2) return 64;
    if (station == 1 && ring == 3) return 32;
    if (station == 2 && ring == 1) return 112;
    if (station == 2 && ring == 2) return 64;
    if (station == 3 && ring == 1) return 96;
    if (station == 3 && ring == 2) return 64;
    if (station == 4 && ring == 1) return 96;
    if (station == 4 && ring == 2) return 64;
    return -1;
  }

  double Detector::Z_mm(int side, int station, int ring, int chamber, int layer) const {
    double z = 0.0;
    // ME1/1 even chambers are closer to IP
    if (station == 1 && ring == 1 && chamber%2 == 0) z = 5834.5 + 22.0*(double)(layer - 1);
    // ME1/1 odd chambers are further from IP
    else
      if (station == 1 && ring == 1 && chamber%2 == 1) z = 6101.5 + 22.0*(double)(layer - 1);
    // All other ME (not ME1/1) - odd chambers mounted to iron, even chambers mounted further from the iron disk
    else
      if (station == 1 && ring == 2 && chamber%2 == 0) z = 6769.48 + (6937.75-6769.48)/7.0*(double)(layer);
    else
      if (station == 1 && ring == 2 && chamber%2 == 1) z = 7043.48 + (7211.75-7043.48)/7.0*(double)(layer);
    else
      if (station == 1 && ring == 3) z = 6867.45 + (7035.72-6867.45)/7.0*(double)(layer);
    else
      if (station == 2 && ring == 1 && chamber%2 == 0) z = 8077.47 + (8245.75-8077.47)/7.0*(double)(layer);
    else
      if (station == 2 && ring == 1 && chamber%2 == 1) z = 8325.48 + (8493.75-8325.48)/7.0*(double)(layer);
    else
      if (station == 2 && ring == 2 && chamber%2 == 0) z = 8077.47 + (8245.75-8077.47)/7.0*(double)(layer);
    else
      if (station == 2 && ring == 2 && chamber%2 == 1) z = 8325.48 + (8493.75-8325.48)/7.0*(double)(layer);
    else
      if (station == 3 && ring == 1 && chamber%2 == 0) z = 9394.25 + (9562.53-9394.25)/7.0*(double)(layer);
    else
      if (station == 3 && ring == 1 && chamber%2 == 1) z = 9146.25 + (9314.52-9146.25)/7.0*(double)(layer);
    else
      if (station == 3 && ring == 2 && chamber%2 == 0) z = 9394.25 + (9562.53-9394.25)/7.0*(double)(layer);
    else
      if (station == 3 && ring == 2 && chamber%2 == 1) z = 9146.25 + (9314.52-9146.25)/7.0*(double)(layer);
    else
      if (station == 4 && ring == 1 && chamber%2 == 0) z = 10289.25 + (10457.53-10289.25)/7.0*(double)(layer);
    else
      if (station == 4 && ring == 1 && chamber%2 == 1) z = 10041.25 + (10209.52-10041.25)/7.0*(double)(layer);
    else
      if (station == 4 && ring == 2 && chamber%2 == 0) z = 10289.25 + (10457.53-10289.25)/7.0*(double)(layer);
    else
      if (station == 4 && ring == 2 && chamber%2 == 1) z = 10041.25 + (10209.52-10041.25)/7.0*(double)(layer);
    z = (double)(SideSign(side)) * z;
    return z;
  }

double Detector::RPin ( int station, int ring ) const {
  double rPin = 100000.0; // Some default large value

  if( station == 1 && ring == 1 ) rPin = 1060.0; // "Pin" = "Chamber's bottom edge" for ME1/1
  if( station == 1 && ring == 2 ) rPin = 2784.9;
  if( station == 1 && ring == 3 ) rPin = 5089.9;
  if( station == 2 && ring == 1 ) rPin = 1438.9;
  if( station == 2 && ring == 2 ) rPin = 3609.9;
  if( station == 3 && ring == 1 ) rPin = 1638.9;
  if( station == 3 && ring == 2 ) rPin = 3609.9;
  if( station == 4 && ring == 1 ) rPin = 1837.9;
  if( station == 4 && ring == 2 ) rPin = 3609.9;

  return rPin;
}

double Detector::PhiDegChamberCenter ( int station, int ring, int chamber ) const {
  double phiDegChamberCenter = 0.0; // Default value

  if( station == 1 && ring == 1 ) phiDegChamberCenter = (chamber - 1)*10.0;
  if( station == 1 && ring == 2 ) phiDegChamberCenter = (chamber - 1)*10.0;
  if( station == 1 && ring == 3 ) phiDegChamberCenter = (chamber - 1)*10.0;
  if( station == 2 && ring == 1 ) phiDegChamberCenter = 5.0 + (chamber - 1)*20.0;
  if( station == 2 && ring == 2 ) phiDegChamberCenter = (chamber - 1)*10.0;
  if( station == 3 && ring == 1 ) phiDegChamberCenter = 5.0 + (chamber - 1)*20.0;
  if( station == 3 && ring == 2 ) phiDegChamberCenter = (chamber - 1)*10.0;
  if( station == 4 && ring == 1 ) phiDegChamberCenter = 5.0 + (chamber - 1)*20.0;
  if( station == 4 && ring == 2 ) phiDegChamberCenter = (chamber - 1)*10.0;

  return phiDegChamberCenter;
}

double Detector::LocalYtoBeam(int side, int station, int ring, int wgroup) const {
  double localYtoBeam = 100000.0; // Some default large value

  localYtoBeam = LocalYtoBeam(side, station, ring, "b", 1, wgroup);

  return localYtoBeam;
}

double Detector::LocalYtoBeam(int side, int station, int ring, const std::string &part, int hstrip, int wgroup) const {
  double localYtoBeam = 100000.0; // Some default large value

  // Special case: ME1/1
  if( station == 1 && ring == 1) {
    int layer = 1;
    // Half strip angle w.r.t. horizontal line
    double localPhiRad = LocalPhiRadHstripToChamberCenter(side, station, ring, part, layer, hstrip);
    // Wire groups incline angle for ME1/1 = 29 degrees
    double angle = (side == 1 ? -1.0 : 1.0) * 29.0;
    double inclineAngleRad = angle * 3.14159 / 180.0;
    if (wgroup == 1 || wgroup == NumberOfWiregroups(1,1)) inclineAngleRad = 0.5 * angle * 3.14159 / 180.0;

    localYtoBeam = ( RPin(station, ring) + dPinToWGCenter_ME1_1[wgroup - 1])
                   / ( 1 - tan(inclineAngleRad)*tan(localPhiRad) );
  }

    // All other chambers
    else if (station == 1 && ring == 2) localYtoBeam = RPin(station, ring) + dPinToWGCenter_ME1_2[wgroup - 1];
    else if (station == 1 && ring == 3) localYtoBeam = RPin(station, ring) + dPinToWGCenter_ME1_3[wgroup - 1];
    else if (station == 2 && ring == 1) localYtoBeam = RPin(station, ring) + dPinToWGCenter_ME2_1[wgroup - 1];
    else if (station == 2 && ring == 2) localYtoBeam = RPin(station, ring) + dPinToWGCenter_ME234_2[wgroup - 1];
    else if (station == 3 && ring == 1) localYtoBeam = RPin(station, ring) + dPinToWGCenter_ME3_1[wgroup - 1];
    else if (station == 3 && ring == 2) localYtoBeam = RPin(station, ring) + dPinToWGCenter_ME234_2[wgroup - 1];
    else if (station == 4 && ring == 1) localYtoBeam = RPin(station, ring) + dPinToWGCenter_ME4_1[wgroup - 1];
    else if (station == 4 && ring == 2) localYtoBeam = RPin(station, ring) + dPinToWGCenter_ME234_2[wgroup - 1];

  return localYtoBeam;
}

int Detector::NumberOfStrips(int station, int ring, const std::string &part) const {
  int nstrips = -1; // Default value

  if( station == 1 && ring == 1 && part == "a") nstrips = 48;
  if( station == 1 && ring == 1 && part == "b") nstrips = 64;
  if( station == 1 && ring == 2) nstrips = 80;
  if( station == 1 && ring == 3) nstrips = 64;
  if( station == 2 && ring == 1) nstrips = 80;
  if( station == 2 && ring == 2) nstrips = 80;
  if( station == 3 && ring == 1) nstrips = 80;
  if( station == 3 && ring == 2) nstrips = 80;
  if( station == 4 && ring == 1) nstrips = 80;
  if( station == 4 && ring == 2) nstrips = 80;

  return nstrips;
}

double Detector::stripDPhiDeg(int station, int ring, const std::string &part) const {
  double stripDPhi = 0; // Default value

  if( station == 1 && ring == 1 && part == "a") stripDPhi = 3.88*1e-3*180/PI; // 3.88 mrad
  if( station == 1 && ring == 1 && part == "b") stripDPhi = 2.96*1e-3*180/PI; // 2.96 mrad
  if( station == 1 && ring == 2) stripDPhi = 10.0/75.0;
  if( station == 1 && ring == 3) stripDPhi = 7.893/64.0;
  if( station == 2 && ring == 1) stripDPhi = 20.0/75.0;
  if( station == 2 && ring == 2) stripDPhi = 10.0/75.0;
  if( station == 3 && ring == 1) stripDPhi = 20.0/75.0;
  if( station == 3 && ring == 2) stripDPhi = 10.0/75.0;
  if( station == 4 && ring == 1) stripDPhi = 20.0/75.0;
  if( station == 4 && ring == 2) stripDPhi = 10.0/75.0;

  return stripDPhi;

}

double Detector::stripStaggerInstripWidth( int station, int ring, int layer ) const {
  double stripStagger = 0.0; // Default value

  if( station == 1 && ring == 1) stripStagger = 0.0;

  if( station == 1 && ring == 2 && (layer == 1 || layer == 3 || layer == 5) ) stripStagger = 0.25;
  if( station == 1 && ring == 2 && (layer == 2 || layer == 4 || layer == 6) ) stripStagger = -0.25;

  if( station == 1 && ring == 3 && (layer == 1 || layer == 3 || layer == 5) ) stripStagger = 0.25;
  if( station == 1 && ring == 3 && (layer == 2 || layer == 4 || layer == 6) ) stripStagger = -0.25;

  if( station == 2 && ring == 1 && (layer == 1 || layer == 3 || layer == 5) ) stripStagger = 0.25;
  if( station == 2 && ring == 1 && (layer == 2 || layer == 4 || layer == 6) ) stripStagger = -0.25;

  if( station == 2 && ring == 2 && (layer == 1 || layer == 3 || layer == 5) ) stripStagger = 0.25;
  if( station == 2 && ring == 2 && (layer == 2 || layer == 4 || layer == 6) ) stripStagger = -0.25;

  if( station == 3 && ring == 1 && (layer == 1 || layer == 3 || layer == 5) ) stripStagger = -0.25;
  if( station == 3 && ring == 1 && (layer == 2 || layer == 4 || layer == 6) ) stripStagger = 0.25;

  if( station == 3 && ring == 2 && (layer == 1 || layer == 3 || layer == 5) ) stripStagger = -0.25;
  if( station == 3 && ring == 2 && (layer == 2 || layer == 4 || layer == 6) ) stripStagger = 0.25;

  if( station == 4 && ring == 1 && (layer == 1 || layer == 3 || layer == 5) ) stripStagger = 0.25;
  if( station == 4 && ring == 1 && (layer == 2 || layer == 4 || layer == 6) ) stripStagger = -0.25;

  if( station == 4 && ring == 2 && (layer == 1 || layer == 3 || layer == 5) ) stripStagger = 0.25;
  if( station == 4 && ring == 2 && (layer == 2 || layer == 4 || layer == 6) ) stripStagger = -0.25;

  return stripStagger;

}

double Detector::LocalPhiDegStripToChamberCenter(int side, int station, int ring, const std::string &part, int layer, int strip) const {
  double localPhi = 0.0;

  localPhi = stripDPhiDeg(station, ring, part)
             * ( (double)strip - (double)NumberOfStrips(station, ring, part)/2.0 - 1.0/2.0 )
       + stripDPhiDeg(station, ring, part) * stripStaggerInstripWidth(station, ring, layer);

  // in the West (positive z) endcap:
  //   strip numbers in stations 1 and 2 are in increasing phi order
  //   strip numbers in stations 3 and 4 are in decreasing phi order
  if (side == 1 && station == 1) {
    localPhi = localPhi;
  }
  if (side == 1 && station == 2) {
    localPhi = localPhi;
  }
  if (side == 1 && station == 3) {
    localPhi = - localPhi;
  }
  if (side == 1 && station == 4) {
    localPhi = - localPhi;
  }

  // in the East (negative z) endcap:
  //   strip numbers in stations 1 and 2 are in decreasing phi order
  //   strip numbers in stations 3 and 4 are in increasing phi order
  if ((side == -1 || side == 2) && station == 1) {
    localPhi = - localPhi;
  }
  if ((side == -1 || side == 2) && station == 2) {
    localPhi = - localPhi;
  }
  if ((side == -1 || side == 2) && station == 3) {
    localPhi = localPhi;
  }
  if ((side == -1 || side == 2) && station == 4) {
    localPhi = localPhi;
  }

  return localPhi;
}

double Detector::LocalPhiDegHstripToChamberCenter(int side, int station, int ring, const std::string &part, int layer, int hstrip) const {
  double localPhi = 0.0;

  localPhi = hstripDPhiDeg(station, ring, part)
             * ( (double)hstrip - (double)NumberOfHalfstrips(station, ring, part)/2.0 - 1.0/2.0 )
       + stripDPhiDeg(station, ring, part) * stripStaggerInstripWidth(station, ring, layer);

  // in the West (positive z) endcap:
  //   strip numbers in stations 1 and 2 are in increasing phi order
  //   strip numbers in stations 3 and 4 are in decreasing phi order
  if (side ==  1 && station == 1) {
    localPhi = localPhi;
  }
  if (side ==  1 && station == 2) {
    localPhi = localPhi;
  }
  if (side ==  1 && station == 3) {
    localPhi = - localPhi;
  }
  if (side ==  1 && station == 4) {
    localPhi = - localPhi;
  }

  // in the East (negative z) endcap:
  //   strip numbers in stations 1 and 2 are in decreasing phi order
  //   strip numbers in stations 3 and 4 are in increasing phi order
  if ((side == -1 || side == 2) && station == 1) {
    localPhi = - localPhi;
  }
  if ((side == -1 || side == 2) && station == 2) {
    localPhi = - localPhi;
  }
  if ((side == -1 || side == 2) && station == 3) {
    localPhi = localPhi;
  }
  if ((side == -1 || side == 2) && station == 4) {
    localPhi = localPhi;
  }

  return localPhi;
}

double Detector::Phi_deg(int side, int station, int ring, const std::string &part, int chamber, int layer, int hstrip) const {
  double phi = 0.0;

  phi =   PhiDegChamberCenter(station, ring, chamber)
        + LocalPhiDegHstripToChamberCenter(side, station, ring, part, layer, hstrip);

  return phi;
}

double Detector::R_mm(int side, int station, int ring, const std::string &part, int layer, int hstrip, int wgroup) const {
  double r = 0.0;

  r =   LocalYtoBeam(side, station, ring, part, hstrip, wgroup)
      / cos( LocalPhiRadHstripToChamberCenter(side, station, ring, part, layer, hstrip) );

  return r;
}

double Detector::X_mm(int side, int station, int ring, const std::string &part, int chamber, int layer, int hstrip, int wgroup) const {
  double x = 0.0;

  x =   R_mm(side, station, ring, part, layer, hstrip, wgroup)
      * cos( Phi_rad(side, station, ring, part, chamber, layer, hstrip) );

  return x;
}

double Detector::Y_mm(int side, int station, int ring, const std::string &part, int chamber, int layer, int hstrip, int wgroup) const {
  double y = 0.0;

  y =   R_mm(side, station, ring, part, layer, hstrip, wgroup)
      * sin( Phi_rad(side, station, ring, part, chamber, layer, hstrip) );

  return y;
}

  double Detector::Theta_rad(int side, int station, int ring, const std::string &part, int chamber, int layer, int hstrip, int wgroup) const {
    double theta = 0.0;
    theta = atan( R_mm(side, station, ring, part, layer, hstrip, wgroup) / Z_mm(side, station, ring, chamber, layer) );
    return theta;
  }

    void Detector::chamberBoundsXY(int side, int station, int ring, int chamber, const std::string& part, int hs, int wg, double& x, double& y) const {
      int LAYER = 1;
      double ME11_STRIP_CUT = 440.0;
      double ME11_LOWER_BOUND = 5.0;
      double ME11_UPPER_DELTA = 10.0;

      if (station == 1 && ring == 1) {

        double pinToWGCenter = dPinToWGCenter_ME1_1[wg - 1];
        if (wg == 1) {
            pinToWGCenter = ME11_LOWER_BOUND;
        }

        // Handle strip cut at ME11_STRIP_CUT
        if (part == "a") {
            if (wg == NumberOfWiregroups(station, ring)) {
                pinToWGCenter = ME11_STRIP_CUT;
            }
        } else {
            if (wg == 1) {
                pinToWGCenter = ME11_STRIP_CUT;
            } else {
                pinToWGCenter += ME11_UPPER_DELTA;
            }
        }

        double localYtoBeam = RPin(station, ring) + pinToWGCenter;

        double r = localYtoBeam /
        cos(LocalPhiRadHstripToChamberCenter(side, station, ring, part, LAYER, hs));
        x = r * cos(Phi_rad(side, station, ring, part, chamber, LAYER, hs));
        y = r * sin(Phi_rad(side, station, ring, part, chamber, LAYER, hs));

      } else {

        x = X_mm(side, station, ring, part, chamber, LAYER, hs, wg);
        y = Y_mm(side, station, ring, part, chamber, LAYER, hs, wg);

      }
    }

    void Detector::chamberBoundsXY(int side, int station, int ring, int chamber, const std::string& part, double* x, double* y) const {
      chamberBoundsXY(side, station, ring, chamber, part, 1, 1, x[0], y[0]);
      chamberBoundsXY(side, station, ring, chamber, part, 1, NumberOfWiregroups(station, ring), x[1], y[1]);
      chamberBoundsXY(side, station, ring, chamber, part, NumberOfHalfstrips(station, ring, part), NumberOfWiregroups(station, ring), x[2], y[2]);
      chamberBoundsXY(side, station, ring, chamber, part, NumberOfHalfstrips(station, ring, part), 1, x[3], y[3]);
    }

}
