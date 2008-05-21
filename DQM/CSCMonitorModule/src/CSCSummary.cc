/*
 * =====================================================================================
 *
 *       Filename:  CSCSummary.cc
 *
 *    Description:  Class CSCSummary implementation
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

#include <iostream>
#include "DQM/CSCMonitorModule/interface/CSCSummary.h"

/**
 * @brief  Constructor
 * @param  
 * @return 
 */
CSCSummary::CSCSummary() {
  Reset();
}

/**
 * @brief  Resets all detector map
 * @return 
 */
void CSCSummary::Reset() {
  CSCAddress adr;
  adr.mask.side = adr.mask.station = false;
  adr.mask.ring = adr.mask.chamber = adr.mask.cfeb = adr.mask.hv = true;
  for (adr.ring = 1; adr.ring <= N_RINGS; adr.ring++) { 
    for (adr.chamber = 1; adr.chamber <= N_CHAMBERS; adr.chamber++) {
       for (adr.cfeb = 1; adr.cfeb <= N_CFEBS; adr.cfeb++) {
          for (adr.hv = 1; adr.hv <= N_HVS; adr.hv++) {
            SetValue(adr, 0);
          }
       }
    }
  }
}

/**
 * @brief  Read Chamber histogram and fill in detector map.
 * @param  h2 Histogram to read
 * @return 
 */
void CSCSummary::ReadChambers(TH2*& h2) {

  if(h2->GetXaxis()->GetXmin() <= 1 && h2->GetXaxis()->GetXmax() >= 36 &&
     h2->GetYaxis()->GetXmin() <= 1 && h2->GetYaxis()->GetXmax() >= 18) {

    CSCAddress adr;

    for(unsigned int x = 1; x <= 36; x++) {
      for(unsigned int y = 1; y <= 18; y++) {
        double z = h2->GetBinContent(x, y);
        if(ChamberCoords(x, y, adr)) {
          SetValue(adr, (z > 0 ? 1 : 0));
        }
      }
    }
  }
}

/**
 * @brief  Write detector map to H1 histogram (linear data)
 * @param  h1 Histogram to write data to
 * @return 
 */
void CSCSummary::Write(TH1*& h1) {
  CSCAddress adr;
  unsigned int bin = 1;

  adr.mask.side = adr.mask.station = adr.mask.ring = adr.mask.chamber = adr.mask.cfeb = adr.mask.hv = true;

  for (adr.side = 1; adr.side <= N_SIDES; adr.side++) { 
    for (adr.station = 1; adr.station <= N_STATIONS; adr.station++) {
       for (adr.ring = 1; adr.ring <= N_RINGS; adr.ring++) { 
          for (adr.chamber = 1; adr.chamber <= N_CHAMBERS; adr.chamber++) {
            for (adr.cfeb = 1; adr.cfeb <= N_CFEBS; adr.cfeb++) {
              for (adr.hv = 1; adr.hv <= N_HVS; adr.hv++) {
                double d = static_cast<double>(GetValue(adr));
                h1->SetBinContent(bin, d);
                bin++;
              }
            }
          }
       }
    }
  }
}

/**
 * @brief  Write detector map to H1 histogram (linear data) for the selected adr.station
 * @param  h1 Histogram to write data to
 * @param  adr.station adr.station number (0-3) to write data for
 * @return 
 */
void CSCSummary::Write(TH1*& h1, const unsigned int station) {
  CSCAddress adr;
  const int station_len = N_RINGS * N_CHAMBERS * N_CFEBS * N_HVS;

  adr.mask.side = adr.mask.station = adr.mask.ring = adr.mask.chamber = adr.mask.cfeb = adr.mask.hv = true;

  if(station < 1 || station > N_STATIONS) return; 

  adr.station = station;

  for (adr.side = 1; adr.side <= N_SIDES; adr.side++) { 
    unsigned int bin = (adr.side - 1) * N_STATIONS * station_len + (adr.station - 1) * station_len + 1;
    for (adr.ring = 1; adr.ring <= N_RINGS; adr.ring++) { 
      for (adr.chamber = 1; adr.chamber <= N_CHAMBERS; adr.chamber++) {
        for (adr.cfeb = 1; adr.cfeb <= N_CFEBS; adr.cfeb++) {
          for (adr.hv = 1; adr.hv <= N_HVS; adr.hv++) {
            double d = static_cast<double>(GetValue(adr));
            h1->SetBinContent(bin, d);
            bin++;
          }
        }
      }
    }
  }
}

/**
 * @brief  Read detector map from H1 histogram (back)
 * @param  h1 Histogram to read detector data from 
 * @return 
 */
void CSCSummary::Read(TH1*& h1) {
  CSCAddress adr;
  unsigned int bin = 1;

  adr.mask.side = adr.mask.station = adr.mask.ring = adr.mask.chamber = adr.mask.cfeb = adr.mask.hv = true;

  for (adr.side = 1; adr.side <= N_SIDES; adr.side++) { 
    for (adr.station = 1; adr.station <= N_STATIONS; adr.station++) {
       for (adr.ring = 1; adr.ring <= N_RINGS; adr.ring++) { 
          for (adr.chamber = 1; adr.chamber <= N_CHAMBERS; adr.chamber++) {
            for (adr.cfeb = 1; adr.cfeb <= N_CFEBS; adr.cfeb++) {
              for (adr.hv = 1; adr.hv <= N_HVS; adr.hv++) {
                double d = h1->GetBinContent(bin);
                int i = static_cast<int>(d);
                SetValue(adr, i);
                bin++;
              }
            }
          }
       }
    }
  }
}

/**
 * @brief  SetValue for the whole of detector
 * @param  value Value to set
 * @return 
 */
void CSCSummary::SetValue(const int value) {
  CSCAddress adr;
  adr.mask.side = adr.mask.station = adr.mask.ring = adr.mask.chamber = adr.mask.cfeb = adr.mask.hv = false;
  SetValue(adr, value);
}

/**
 * @brief  Set value recursivelly by following the supplied address
 * @param  adr CSCAddress to be updated
 * @param  value Value to be set
 * @return 
 */
void CSCSummary::SetValue(CSCAddress adr, const int value) {

  if (!adr.mask.side) {
    adr.mask.side = true;
    for (adr.side = 1; adr.side <= N_SIDES; adr.side++) SetValue(adr, value);
    return;
  }

  if (!adr.mask.station) {
    adr.mask.station = true;
    for (adr.station = 1; adr.station <= N_STATIONS; adr.station++) SetValue(adr, value);
    return;
  }

  if (!adr.mask.ring) {
    adr.mask.ring = true;
    for (adr.ring = 1; adr.ring <= NumberOfRings(adr.station); adr.ring++) SetValue(adr, value);
    return;
  }

  if (!adr.mask.chamber) {
    adr.mask.chamber = true;
    for (adr.chamber = 1; adr.chamber <= NumberOfChambers(adr.station, adr.ring); adr.chamber++) SetValue(adr, value);
    return;
  }

  if (!adr.mask.cfeb) {
    adr.mask.cfeb = true;
    for (adr.cfeb = 1; adr.cfeb <= NumberOfChamberCFEBs(adr.station, adr.ring); adr.cfeb++) SetValue(adr, value);
    return;
  }

  if (!adr.mask.hv) {
    adr.mask.hv = true;
    for (adr.hv = 1; adr.hv <= NumberOfChamberHVs(adr.station, adr.ring); adr.hv++) SetValue(adr, value);
    return;
  }

  if( adr.side > 0 && adr.side <= N_SIDES && adr.station > 0 && adr.station <= N_STATIONS && 
      adr.ring > 0 && adr.ring <= N_RINGS && adr.chamber > 0 && adr.chamber <= N_CHAMBERS && 
      adr.cfeb > 0 && adr.cfeb <= N_CFEBS && adr.hv > 0 && adr.hv <= N_HVS) {

    map[adr.side - 1][adr.station - 1][adr.ring - 1][adr.chamber - 1][adr.cfeb - 1][adr.hv - 1] = value;

  }

}

/**
 * @brief  Get efficiency of the whole detector
 * @param  
 * @return Detector efficiency rate (0..1)
 */
const double CSCSummary::GetEfficiency() {
  CSCAddress adr;
  adr.mask.side = adr.mask.station = adr.mask.ring = adr.mask.chamber = adr.mask.cfeb = adr.mask.hv = false;
  return GetEfficiency(adr);
}

/**
 * @brief  Get efficiency of the detector part supplied by the address
 * @param  adr Address to watch efficiency for
 * @return Subdetector efficiency rate (0..1)
 */
const double CSCSummary::GetEfficiency(CSCAddress adr) { 
  double sum = 0.0;

  if (!adr.mask.side) {
    adr.mask.side = true;
    for (adr.side = 1; adr.side <= N_SIDES; adr.side++) sum += GetEfficiency(adr);
    return sum / N_SIDES;
  }

  if (!adr.mask.station) {
    adr.mask.station = true;
    for (adr.station = 1; adr.station <= N_STATIONS; adr.station++) sum += GetEfficiency(adr);
    return sum / N_STATIONS;
  }

  if (!adr.mask.ring) {
    adr.mask.ring = true;
    for (adr.ring = 1; adr.ring <= NumberOfRings(adr.station); adr.ring++) sum += GetEfficiency(adr);
    return sum / NumberOfRings(adr.station);
  }

  if (!adr.mask.chamber) {
    adr.mask.chamber = true;
    for (adr.chamber = 1; adr.chamber <= NumberOfChambers(adr.station, adr.ring); adr.chamber++) sum += GetEfficiency(adr);
    return sum / NumberOfChambers(adr.station, adr.ring);
  }

  if (!adr.mask.cfeb) {
    adr.mask.cfeb = true;
    for (adr.cfeb = 1; adr.cfeb <= NumberOfChamberCFEBs(adr.station, adr.ring); adr.cfeb++) sum += GetEfficiency(adr);
    return sum / NumberOfChamberCFEBs(adr.station, adr.ring);
  }

  if (!adr.mask.hv) {
    adr.mask.hv = true;
    for (adr.hv = 1; adr.hv <= NumberOfChamberHVs(adr.station, adr.ring); adr.hv++) sum += GetEfficiency(adr);
    return sum / NumberOfChamberHVs(adr.station, adr.ring);
  }

  if (GetValue(adr) > 0) return 1.0;

  return 0.0;
}

/**
 * @brief  Get value of some address (address must be fully filled! otherwise function returns -1)
 * @param  adr Address of atomic element to return value from
 * @return Value of the requested element
 */
const int CSCSummary::GetValue(CSCAddress adr) {
  if( adr.mask.side && adr.mask.station && adr.mask.ring && 
      adr.mask.chamber && adr.mask.cfeb && adr.mask.hv &&
      adr.side > 0 && adr.side <= N_SIDES && 
      adr.station > 0 && adr.station <= N_STATIONS && 
      adr.ring > 0 && adr.ring <= N_RINGS && 
      adr.chamber > 0 && adr.chamber <= N_CHAMBERS && 
      adr.cfeb > 0 && adr.cfeb <= N_CFEBS && 
      adr.hv > 0 && adr.hv <= N_HVS) {
    return map[adr.side - 1][adr.station - 1][adr.ring - 1][adr.chamber - 1][adr.cfeb - 1][adr.hv - 1];
  }
  return -1;
}

/**
 * @brief  Calculate CSCAddress from CSCChamberMap histogram coordinates 
 * @param  x X coordinate of histogram
 * @param  y Y coordinate of histogram
 * @param  adr CSCAddress to be filled in and returned
 * @return true if address was found and filled, false - otherwise
 */
const bool CSCSummary::ChamberCoords(const unsigned int x, const unsigned int y, CSCAddress& adr) {

  if( x < 1 || x > 36 || y < 1 || y > 18) return false;

  adr.mask.side = adr.mask.station = adr.mask.ring = adr.mask.chamber = true;
  adr.mask.cfeb = adr.mask.hv = false;

  if ( y < 10 ) adr.side = 2;
  else adr.side = 1;

  adr.chamber = x;

  if (y == 1 || y == 18) {
    adr.station = 4;
    adr.ring    = 2;
  } else
  if (y == 2 || y == 17) {
    adr.station = 4;
    adr.ring    = 1;
  } else
  if (y == 3 || y == 16) {
    adr.station = 3;
    adr.ring    = 2;
  } else
  if (y == 4 || y == 15) {
    adr.station = 3;
    adr.ring    = 1;
  } else
  if (y == 5 || y == 14) {
    adr.station = 2;
    adr.ring    = 2;
  } else
  if (y == 6 || y == 13) {
    adr.station = 2;
    adr.ring    = 1;
  } else
  if (y == 7 || y == 12) {
    adr.station = 1;
    adr.ring    = 3;
  } else
  if (y == 8 || y == 11) {
    adr.station = 1;
    adr.ring    = 2;
  } else
  if (y == 9 || y == 10) {
    adr.station = 1;
    adr.ring    = 1;
  }

  return true;

}

/**
 * @brief  Returns the number of rings for the given station
 * @param  station Station number (1, 2, 3, 4)
 * @return number of rings for the given station
 */
const unsigned int CSCSummary::NumberOfRings(const unsigned int station) {
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
const unsigned int CSCSummary::NumberOfChambers(const unsigned int station, const unsigned int ring) {
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
const unsigned int CSCSummary::NumberOfChamberCFEBs(const unsigned int station, const unsigned int ring) {
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
const unsigned int CSCSummary::NumberOfChamberHVs(const unsigned int station, const unsigned int ring) {
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
void CSCSummary::PrintAddress(const CSCAddress& adr) {

  std::cout << "Side (" << std::boolalpha << adr.mask.side << ")"; 
  if (adr.mask.side) std::cout << adr.side;

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

const bool CSCSummary::Iterator(unsigned int& i, CSCAddress& adr, const CSCAddressMask& mask) {
  
  return 0.0;
}

const unsigned int CSCSummary::NumberOfElements() {
  CSCAddress adr;
  adr.mask.side = adr.mask.station = adr.mask.ring = adr.mask.chamber = adr.mask.cfeb = adr.mask.hv = false;
  return NumberOfElements(adr);
}

const unsigned int CSCSummary::NumberOfElements(const CSCAddress adr) {
  unsigned int n = 0;

  if (!adr.mask.side) {
    adr.mask.side = true;
    for (adr.side = 1; adr.side <= N_SIDES; adr.side++) n += NumberOfElements(adr);
    return n;
  }

  if (!adr.mask.station) {
    adr.mask.station = true;
    for (adr.station = 1; adr.station <= N_STATIONS; adr.station++) n += NumberOfElements(adr);
    return n;
  }

  if (!adr.mask.ring) {
    adr.mask.ring = true;
    for (adr.ring = 1; adr.ring <= NumberOfRings(adr.station); adr.ring++) n += NumberOfElements(adr);
    return n;
  }

  if (!adr.mask.chamber) {
    return NumberOfChambers(adr.station, adr.ring) * NumberOfChamberCFEBs(adr.station, adr.ring) * NumberOfChamberHVs(adr.station, adr.ring);
  }

  if (!adr.mask.cfeb) {
    return NumberOfChamberCFEBs(adr.station, adr.ring) * NumberOfChamberHVs(adr.station, adr.ring);
  }

  if (!adr.mask.hv) {
    return NumberOfChamberHVs(adr.station, adr.ring);
  }

  return 1;

}

