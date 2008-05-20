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
  SetValue(V_NULL);
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
            SetValue(side, station, ring, chamber, V_TRUE);
          } else {
            SetValue(side, station, ring, chamber, V_FALSE);
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
  for (unsigned int side = 0; side < N_SIDES; side++) { 
    for (unsigned int station = 0; station < N_STATIONS; station++) {
       for (unsigned int ring = 0; ring < N_RINGS; ring++) { 
          for (unsigned int chamber = 0; chamber < N_CHAMBERS; chamber++) {
            for (unsigned int cfeb = 0; cfeb < N_CFEBS; cfeb++) {
              for (unsigned int hv = 0; hv < N_HVS; hv++) {
                double d = static_cast<double>(map[side][station][ring][chamber][cfeb][hv]);
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
  if(station < 0 || station > (N_STATIONS - 1)) return; 
  for (unsigned int side = 0; side < N_SIDES; side++) { 
    unsigned int bin = side * N_STATIONS * len + station * len + 1;
    for (unsigned int ring = 0; ring < N_RINGS; ring++) { 
      for (unsigned int chamber = 0; chamber < N_CHAMBERS; chamber++) {
        for (unsigned int cfeb = 0; cfeb < N_CFEBS; cfeb++) {
          for (unsigned int hv = 0; hv < N_HVS; hv++) {
            double d = static_cast<double>(map[side][station][ring][chamber][cfeb][hv]);
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
  for (unsigned int side = 0; side < N_SIDES; side++) { 
    for (unsigned int station = 0; station < N_STATIONS; station++) {
       for (unsigned int ring = 0; ring < N_RINGS; ring++) { 
          for (unsigned int chamber = 0; chamber < N_CHAMBERS; chamber++) {
            for (unsigned int cfeb = 0; cfeb < N_CFEBS; cfeb++) {
              for (unsigned int hv = 0; hv < N_HVS; hv++) {
                double d = h1->GetBinContent(bin);
                map[side][station][ring][chamber][cfeb][hv] = static_cast<int>(d);
                bin++;
              }
            }
          }
       }
    }
  }
}

void CSCSummary::SetValue(const int value) {

  for (unsigned int side = 0; side < N_SIDES; side++) {
    SetValue(side, value);
  }

}

void CSCSummary::SetValue(
    const unsigned int side, 
    const int value) {

  for (unsigned int station = 0; station < N_STATIONS; station++) {
    SetValue(side, station, value);
  }

}

void CSCSummary::SetValue(
    const unsigned int side, 
    const unsigned int station, 
    const int value) {

  for (unsigned int ring = 0; ring < N_RINGS; ring++) {
    SetValue(side, station, ring, value);
  }

}

void CSCSummary::SetValue(
    const unsigned int side, 
    const unsigned int station, 
    const unsigned int ring, 
    const int value) {

  for (unsigned int chamber = 0; chamber < N_CHAMBERS; chamber++) {
    SetValue(side, station, ring, chamber, value);
  }

}

void CSCSummary::SetValue(
    const unsigned int side, 
    const unsigned int station, 
    const unsigned int ring, 
    const unsigned int chamber, 
    const int value) {

  for (unsigned int cfeb = 0; cfeb < N_CFEBS; cfeb++) {
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

  for (unsigned int hv = 0; hv < N_HVS; hv++) {
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

  if(side < N_SIDES && station < N_STATIONS && ring < N_RINGS && chamber < N_CHAMBERS && cfeb < N_CFEBS && hv < N_HVS) {
    map[side][station][ring][chamber][cfeb][hv] = value;
  }

}

const double CSCSummary::GetEfficiency() {
  double sum = 0.0;
  for (unsigned int side = 0; side < N_SIDES; side++) {
    sum += GetEfficiency(side);
  }
  return sum / N_SIDES;
}

const double CSCSummary::GetEfficiency(
    const unsigned int side) { 
  double sum = 0.0;
  for (unsigned int station = 0; station < N_STATIONS; station++) {
    sum += GetEfficiency(side, station);
  }
  return sum / N_STATIONS;
}

const double CSCSummary::GetEfficiency(
    const unsigned int side, 
    const unsigned int station) {
  double sum = 0.0;
  for (unsigned int ring = 0; ring < N_RINGS; ring++) {
    sum += GetEfficiency(side, station, ring);
  }
  return sum / N_RINGS;
}

const double CSCSummary::GetEfficiency(
    const unsigned int side, 
    const unsigned int station, 
    const unsigned int ring) {
  double sum = 0.0;
  for (unsigned int chamber = 0; chamber < N_CHAMBERS; chamber++) {
    sum += GetEfficiency(side, station, ring, chamber);
  }
  return sum / N_CHAMBERS;
}

const double CSCSummary::GetEfficiency(
    const unsigned int side, 
    const unsigned int station, 
    const unsigned int ring, 
    const unsigned int chamber) {
  double sum = 0.0;
  for (unsigned int cfeb = 0; cfeb < N_CFEBS; cfeb++) {
    sum += GetEfficiency(side, station, ring, chamber, cfeb);
  }
  return sum / N_CFEBS;
}

const double CSCSummary::GetEfficiency(
    const unsigned int side, 
    const unsigned int station, 
    const unsigned int ring, 
    const unsigned int chamber, 
    const unsigned int cfeb) {
  double sum = 0.0;
  for (unsigned int hv = 0; hv < N_HVS; hv++) {
    sum += GetEfficiency(side, station, ring, chamber, cfeb, hv);
  }
  return sum / N_HVS;
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
  if(side < N_SIDES && station < N_STATIONS && ring < N_RINGS && chamber < N_CHAMBERS && cfeb < N_CFEBS && hv < N_HVS) {
    return map[side][station][ring][chamber][cfeb][hv];
  }
  return V_NULL;
}

bool CSCSummary::ChamberCoords(const unsigned int x, const unsigned int y,
                              unsigned int& side,
                              unsigned int& station,
                              unsigned int& ring,
                              unsigned int& chamber) {

  if( x < 1 || x > 36 || y < 1 || y > 18) return false;

  if ( y < 10 ) side = 1;
  else side = 0;

  chamber = x - 1;

  switch(y) {
    case 1:
      station = 3;
      ring    = 1;
      break;
    case 2:
      station = 3;
      ring    = 0;
      break;
    case 3:
      station = 3;
      ring    = 1;
      break;
    case 4:
      station = 2;
      ring    = 0;
      break;
    case 5:
      station = 1;
      ring    = 1;
      break;
    case 6:
      station = 1;
      ring    = 0;
      break;
    case 7:
      station = 0;
      ring    = 2;
      break;
    case 8:
      station = 0;
      ring    = 1;
      break;
    case 9:
      station = 0;
      ring    = 0;
      break;
    case 10:
      station = 0;
      ring    = 0;
      break;
    case 11:
      station = 0;
      ring    = 1;
      break;
    case 12:
      station = 0;
      ring    = 2;
      break;
    case 13:
      station = 1;
      ring    = 0;
      break;
    case 14:
      station = 1;
      ring    = 1;
      break;
    case 15:
      station = 2;
      ring    = 0;
      break;
    case 16:
      station = 2;
      ring    = 1;
      break;
    case 17:
      station = 3;
      ring    = 0;
      break;
    case 18:
      station = 3;
      ring    = 1;
      break;
  }

  return true;

}
