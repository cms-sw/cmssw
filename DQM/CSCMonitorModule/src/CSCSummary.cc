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
  for (unsigned int side = 1; side <= N_SIDES; side++) { 
    for (unsigned int station = 1; station <= N_STATIONS; station++) {
       for (unsigned int ring = 1; ring <= N_RINGS; ring++) { 
          for (unsigned int chamber = 1; chamber <= N_CHAMBERS; chamber++) {
            for (unsigned int cfeb = 1; cfeb <= N_CFEBS; cfeb++) {
              for (unsigned int hv = 1; hv <= N_HVS; hv++) {
                SetValue(side, station, ring, chamber, cfeb, hv, 0);
              }
            }
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

    unsigned int side, station, ring, chamber;
    for(unsigned int x = 1; x <= 36; x++) {
      for(unsigned int y = 1; y <= 18; y++) {
        double z = h2->GetBinContent(x, y);
        if(ChamberCoords(x, y, side, station, ring, chamber)) {
          if(z > 0) {
            SetValue(side, station, ring, chamber, 1);
          } else {
            SetValue(side, station, ring, chamber, 0);
          }
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
  unsigned int bin = 1;
  for (unsigned int side = 1; side <= N_SIDES; side++) { 
    for (unsigned int station = 1; station <= N_STATIONS; station++) {
       for (unsigned int ring = 1; ring <= N_RINGS; ring++) { 
          for (unsigned int chamber = 1; chamber <= N_CHAMBERS; chamber++) {
            for (unsigned int cfeb = 1; cfeb <= N_CFEBS; cfeb++) {
              for (unsigned int hv = 1; hv <= N_HVS; hv++) {
                int i = GetValue(side, station, ring, chamber, cfeb, hv);
                double d = static_cast<double>(i);
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
 * @brief  Write detector map to H1 histogram (linear data) for the selected station
 * @param  h1 Histogram to write data to
 * @param  station Station number (0-3) to write data for
 * @return 
 */
void CSCSummary::Write(TH1*& h1, const unsigned int station) {
  const int len = N_RINGS * N_CHAMBERS * N_CFEBS * N_HVS;
  if(station < 1 || station > N_STATIONS) return; 
  for (unsigned int side = 1; side <= N_SIDES; side++) { 
    unsigned int bin = (side - 1) * N_STATIONS * len + (station - 1) * len + 1;
    for (unsigned int ring = 1; ring <= N_RINGS; ring++) { 
      for (unsigned int chamber = 1; chamber <= N_CHAMBERS; chamber++) {
        for (unsigned int cfeb = 1; cfeb <= N_CFEBS; cfeb++) {
          for (unsigned int hv = 1; hv <= N_HVS; hv++) {
            double d = static_cast<double>(GetValue(side, station, ring, chamber, cfeb, hv));
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
  unsigned int bin = 1;
  for (unsigned int side = 1; side <= N_SIDES; side++) { 
    for (unsigned int station = 1; station <= N_STATIONS; station++) {
       for (unsigned int ring = 1; ring <= N_RINGS; ring++) { 
          for (unsigned int chamber = 1; chamber <= N_CHAMBERS; chamber++) {
            for (unsigned int cfeb = 1; cfeb <= N_CFEBS; cfeb++) {
              for (unsigned int hv = 1; hv <= N_HVS; hv++) {
                double d = h1->GetBinContent(bin);
                int i = static_cast<int>(d);
                SetValue(side, station, ring, chamber, cfeb, hv, i);
                bin++;
              }
            }
          }
       }
    }
  }
}

void CSCSummary::SetValue(const int value) {
  for (unsigned int side = 1; side <= N_SIDES; side++) {
    SetValue(side, value);
  }
}

void CSCSummary::SetValue(
    const unsigned int side, 
    const int value) {
  for (unsigned int station = 1; station <= N_STATIONS; station++) {
    SetValue(side, station, value);
  }
}

void CSCSummary::SetValue(
    const unsigned int side, 
    const unsigned int station, 
    const int value) {
  for (unsigned int ring = 1; ring <= NumberOfRings(station); ring++) {
    SetValue(side, station, ring, value);
  }
}

void CSCSummary::SetValue(
    const unsigned int side, 
    const unsigned int station, 
    const unsigned int ring, 
    const int value) {
  for (unsigned int chamber = 1; chamber <= NumberOfChambers(station, ring); chamber++) {
    SetValue(side, station, ring, chamber, value);
  }
}

void CSCSummary::SetValue(
    const unsigned int side, 
    const unsigned int station, 
    const unsigned int ring, 
    const unsigned int chamber, 
    const int value) {
  for (unsigned int cfeb = 1; cfeb <= NumberOfChamberCFEBs(station, ring); cfeb++) {
    SetValue(side, station, ring, chamber, cfeb, value);
  }
}

void CSCSummary::SetValue(
    const unsigned int side, 
    const unsigned int station, 
    const unsigned int ring, 
    const unsigned int chamber, 
    const unsigned int cfeb, 
    const int value) {
  for (unsigned int hv = 1; hv <= NumberOfChamberHVs(station, ring); hv++) {
    SetValue(side, station, ring, chamber, cfeb, hv, value);
  }
}

void CSCSummary::SetValue(
    const unsigned int side, 
    const unsigned int station, 
    const unsigned int ring, 
    const unsigned int chamber, 
    const unsigned int cfeb, 
    const unsigned int hv, 
    const int value) {
  if( side > 0 && side <= N_SIDES && 
      station > 0 && station <= N_STATIONS && 
      ring > 0 && ring <= N_RINGS && 
      chamber > 0 && chamber <= N_CHAMBERS && 
      cfeb > 0 && cfeb <= N_CFEBS && 
      hv > 0 && hv <= N_HVS) {
    map[side - 1][station - 1][ring - 1][chamber - 1][cfeb - 1][hv - 1] = value;
  }
}

const double CSCSummary::GetEfficiency() {
  double sum = 0.0;
  for (unsigned int side = 1; side <= N_SIDES; side++) {
    sum += GetEfficiency(side);
  }
  return sum / N_SIDES;
}

const double CSCSummary::GetEfficiency(
    const unsigned int side) { 
  double sum = 0.0;
  for (unsigned int station = 1; station <= N_STATIONS; station++) {
    sum += GetEfficiency(side, station);
  }
  return sum / N_STATIONS;
}

const double CSCSummary::GetEfficiency(
    const unsigned int side, 
    const unsigned int station) {
  double sum = 0.0;
  for (unsigned int ring = 1; ring <= NumberOfRings(station); ring++) {
    sum += GetEfficiency(side, station, ring);
  }
  return sum / NumberOfRings(station);
}

const double CSCSummary::GetEfficiency(
    const unsigned int side, 
    const unsigned int station, 
    const unsigned int ring) {
  double sum = 0.0;
  for (unsigned int chamber = 1; chamber <= NumberOfChambers(station, ring); chamber++) {
    sum += GetEfficiency(side, station, ring, chamber);
  }
  return sum / NumberOfChambers(station, ring);
}

const double CSCSummary::GetEfficiency(
    const unsigned int side, 
    const unsigned int station, 
    const unsigned int ring, 
    const unsigned int chamber) {
  double sum = 0.0;
  for (unsigned int cfeb = 1; cfeb <= NumberOfChamberCFEBs(station, ring); cfeb++) {
    sum += GetEfficiency(side, station, ring, chamber, cfeb);
  }
  return sum / NumberOfChamberCFEBs(station, ring);
}

const double CSCSummary::GetEfficiency(
    const unsigned int side, 
    const unsigned int station, 
    const unsigned int ring, 
    const unsigned int chamber, 
    const unsigned int cfeb) {
  double sum = 0.0;
  for (unsigned int hv = 1; hv <= NumberOfChamberHVs(station, ring); hv++) {
    sum += GetEfficiency(side, station, ring, chamber, cfeb, hv);
  }
  return sum / NumberOfChamberHVs(station, ring);
}

const double CSCSummary::GetEfficiency(
    const unsigned int side, 
    const unsigned int station, 
    const unsigned int ring, 
    const unsigned int chamber, 
    const unsigned int cfeb, 
    const unsigned int hv) {
  int i = GetValue(side, station, ring, chamber, cfeb, hv);
  if (i > 0) return 1.0;
  return 0.0;
}

const int CSCSummary::GetValue(
    const unsigned int side, 
    const unsigned int station, 
    const unsigned int ring, 
    const unsigned int chamber, 
    const unsigned int cfeb, 
    const unsigned int hv) {
  if( side > 0 && side <= N_SIDES && 
      station > 0 && station <= N_STATIONS && 
      ring > 0 && ring <= N_RINGS && 
      chamber > 0 && chamber <= N_CHAMBERS && 
      cfeb > 0 && cfeb <= N_CFEBS && 
      hv > 0 && hv <= N_HVS) {
    return map[side - 1][station - 1][ring - 1][chamber - 1][cfeb - 1][hv - 1];
  }
  return -1;
}

const bool CSCSummary::ChamberCoords(const unsigned int x, const unsigned int y,
                              unsigned int& side,
                              unsigned int& station,
                              unsigned int& ring,
                              unsigned int& chamber) {

  if( x < 1 || x > 36 || y < 1 || y > 18) return false;

  if ( y < 10 ) side = 2;
  else side = 1;

  chamber = x;

  if (y == 1 || y == 18) {
    station = 4;
    ring    = 2;
  } else
  if (y == 2 || y == 17) {
    station = 4;
    ring    = 1;
  } else
  if (y == 3 || y == 16) {
    station = 3;
    ring    = 2;
  } else
  if (y == 4 || y == 15) {
    station = 3;
    ring    = 1;
  } else
  if (y == 5 || y == 14) {
    station = 2;
    ring    = 2;
  } else
  if (y == 6 || y == 13) {
    station = 2;
    ring    = 1;
  } else
  if (y == 7 || y == 12) {
    station = 1;
    ring    = 3;
  } else
  if (y == 8 || y == 11) {
    station = 1;
    ring    = 2;
  } else
  if (y == 9 || y == 10) {
    station = 1;
    ring    = 1;
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

