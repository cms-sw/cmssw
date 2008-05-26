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
void CSCSummary::ReadChambers(TH2*& h2, const double threshold) {

  if(h2->GetXaxis()->GetXmin() <= 1 && h2->GetXaxis()->GetXmax() >= 36 &&
     h2->GetYaxis()->GetXmin() <= 1 && h2->GetYaxis()->GetXmax() >= 18) {

    CSCAddress adr;

    for(unsigned int x = 1; x <= 36; x++) {
      for(unsigned int y = 1; y <= 18; y++) {
        double z = h2->GetBinContent(x, y);
        if(ChamberCoords(x, y, adr)) {
          SetValue(adr, (z >= threshold ? 1 : 0));
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
    for (adr.ring = 1; adr.ring <= detector.NumberOfRings(adr.station); adr.ring++) SetValue(adr, value);
    return;
  }

  if (!adr.mask.chamber) {
    adr.mask.chamber = true;
    for (adr.chamber = 1; adr.chamber <= detector.NumberOfChambers(adr.station, adr.ring); adr.chamber++) SetValue(adr, value);
    return;
  }

  if (!adr.mask.cfeb) {
    adr.mask.cfeb = true;
    for (adr.cfeb = 1; adr.cfeb <= detector.NumberOfChamberCFEBs(adr.station, adr.ring); adr.cfeb++) SetValue(adr, value);
    return;
  }

  if (!adr.mask.hv) {
    adr.mask.hv = true;
    for (adr.hv = 1; adr.hv <= detector.NumberOfChamberHVs(adr.station, adr.ring); adr.hv++) SetValue(adr, value);
    return;
  }

  if( adr.side > 0 && adr.side <= N_SIDES && adr.station > 0 && adr.station <= N_STATIONS && 
      adr.ring > 0 && adr.ring <= N_RINGS && adr.chamber > 0 && adr.chamber <= N_CHAMBERS && 
      adr.cfeb > 0 && adr.cfeb <= N_CFEBS && adr.hv > 0 && adr.hv <= N_HVS) {

    map[adr.side - 1][adr.station - 1][adr.ring - 1][adr.chamber - 1][adr.cfeb - 1][adr.hv - 1] = value;

  }

}

const bool CSCSummary::IsPhysicsReady(const float xmin, const float xmax, const float ymin, const float ymax) const {

  float xpmin = (xmin < xmax ? xmin : xmax);
  float xpmax = (xmax > xmin ? xmax : xmin);
  float ypmin = (ymin < ymax ? ymin : ymax);
  float ypmax = (ymax > ymin ? ymax : ymin);

  if (xmin >= -1.0 && xmax <= 1.0) return false; 

  unsigned int i = 0, sum = 0;
  CSCAddress adr;
  const CSCAddressBox *box;

  adr.mask.ring = adr.mask.chamber = adr.mask.cfeb = adr.mask.hv = false;
  adr.mask.side = adr.mask.station = true;
  adr.side = (xmin > 0 ? 1 : 2);

  for (adr.station = 1; adr.station <= N_STATIONS; adr.station++) {

    while(detector.NextAddressBox(i, box, adr)) {
      
      float xboxmin = (box->xmin < box->xmax ? box->xmin : box->xmax);
      float xboxmax = (box->xmax > box->xmin ? box->xmax : box->xmin);
      float yboxmin = (box->ymin < box->ymax ? box->ymin : box->ymax);
      float yboxmax = (box->ymax > box->ymin ? box->ymax : box->ymin);

      if ((xpmin < xboxmin && xpmax <= xboxmin) || (xpmin >= xboxmax && xpmax > xboxmax)) continue;
      if ((ypmin < yboxmin && ypmax <= yboxmin) || (ypmin >= yboxmax && ypmax > yboxmax)) continue;

      //std::cout << "Request: " << xmin << ", " << xmax << ", " << ymin << ", " << ymax << std::endl;
      //std::cout << "Respons: " << box->xmin << ", " << box->xmax << ", " << box->ymin << ", " << box->ymax << std::endl;
      //detector.PrintAddress(box->adr);

      if (GetValue(box->adr) > 0) {
        sum++;
        break;
      }

    }

    if (sum > 2) return true;

  }

  return false;

}

/**
 * @brief  Get efficiency of the whole detector
 * @param  
 * @return Detector efficiency rate (0..1)
 */
const double CSCSummary::GetEfficiencyHW() const {
  CSCAddress adr;
  adr.mask.side = adr.mask.station = adr.mask.ring = adr.mask.chamber = adr.mask.cfeb = adr.mask.hv = false;
  return GetEfficiencyHW(adr);
}

/**
 * @brief  Get efficiency of the detector part supplied by the address
 * @param  adr Address to watch efficiency for
 * @return Subdetector efficiency rate (0..1)
 */
const double CSCSummary::GetEfficiencyHW(CSCAddress adr) const { 
  double sum = 0.0;

  if (!adr.mask.side) {
    adr.mask.side = true;
    for (adr.side = 1; adr.side <= N_SIDES; adr.side++) sum += GetEfficiencyHW(adr);
    return sum / N_SIDES;
  }

  if (!adr.mask.station) {
    adr.mask.station = true;
    for (adr.station = 1; adr.station <= N_STATIONS; adr.station++) sum += GetEfficiencyHW(adr);
    return sum / N_STATIONS;
  } 

  if (!adr.mask.ring) {
    adr.mask.ring = true;
    for (adr.ring = 1; adr.ring <= detector.NumberOfRings(adr.station); adr.ring++) sum += GetEfficiencyHW(adr);
    return sum / detector.NumberOfRings(adr.station);
  }

  if (!adr.mask.chamber) {
    adr.mask.chamber = true;
    for (adr.chamber = 1; adr.chamber <= detector.NumberOfChambers(adr.station, adr.ring); adr.chamber++) sum += GetEfficiencyHW(adr);
    return sum / detector.NumberOfChambers(adr.station, adr.ring);
  }

  if (!adr.mask.cfeb) {
    adr.mask.cfeb = true;
    for (adr.cfeb = 1; adr.cfeb <= detector.NumberOfChamberCFEBs(adr.station, adr.ring); adr.cfeb++) sum += GetEfficiencyHW(adr);
    return sum / detector.NumberOfChamberCFEBs(adr.station, adr.ring);
  }

  if (!adr.mask.hv) {
    adr.mask.hv = true;
    for (adr.hv = 1; adr.hv <= detector.NumberOfChamberHVs(adr.station, adr.ring); adr.hv++) sum += GetEfficiencyHW(adr);
    return sum / detector.NumberOfChamberHVs(adr.station, adr.ring);
  }

  if (GetValue(adr) > 0) return 1.0;

  return 0.0;

}

const double CSCSummary::GetEfficiencyArea(CSCAddress adr) const {
  double all_area = 1;

  if(adr.side == adr.ring == adr.chamber == adr.cfeb == adr.hv == false)
    all_area = detector.Area(adr.station);
  else
    all_area = detector.Area(adr);

  double rep_area = GetReportingArea(adr);
  return rep_area / all_area;
}

const double CSCSummary::GetReportingArea(CSCAddress adr) const { 
  double sum = 0.0;

  if (!adr.mask.side) {
    adr.mask.side = true;
    for (adr.side = 1; adr.side <= N_SIDES; adr.side++) sum += GetReportingArea(adr);
    return sum;
  }

  if (!adr.mask.station) {
    adr.mask.station = true;
    for (adr.station = 1; adr.station <= N_STATIONS; adr.station++) sum += GetReportingArea(adr);
    return sum;
  } 

  if (!adr.mask.ring) {
    adr.mask.ring = true;
    for (adr.ring = 1; adr.ring <= detector.NumberOfRings(adr.station); adr.ring++) sum += GetReportingArea(adr);
    return sum;
  }

  if (!adr.mask.chamber) {
    adr.mask.chamber = true;
    for (adr.chamber = 1; adr.chamber <= detector.NumberOfChambers(adr.station, adr.ring); adr.chamber++) sum += GetReportingArea(adr);
    return sum;
  }

  if (!adr.mask.cfeb) {
    adr.mask.cfeb = true;
    for (adr.cfeb = 1; adr.cfeb <= detector.NumberOfChamberCFEBs(adr.station, adr.ring); adr.cfeb++) sum += GetReportingArea(adr);
    return sum;
  }

  if (!adr.mask.hv) {
    adr.mask.hv = true;
    for (adr.hv = 1; adr.hv <= detector.NumberOfChamberHVs(adr.station, adr.ring); adr.hv++) sum += GetReportingArea(adr);
    return sum;
  }

  if (GetValue(adr) > 0) return detector.Area(adr);

  return 0.0;

}
/**
 * @brief  Get value of some address (address must be fully filled! otherwise function returns -1)
 * @param  adr Address of atomic element to return value from
 * @return Value of the requested element
 */
const int CSCSummary::GetValue(const CSCAddress& adr) const {
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
const bool CSCSummary::ChamberCoords(const unsigned int x, const unsigned int y, CSCAddress& adr) const {

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

