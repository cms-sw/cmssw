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
#include "DQM/CSCMonitorModule/interface/CSCMonitorModule.h"

/**
 * @brief  Constructor
 * @param  
 * @return 
 */
CSCSummary::CSCSummary() {
  Reset();
}

/**
 * @brief  Destructor
 * @param  
 * @return 
 */
CSCSummary::~CSCSummary() {
  // Lets delete masked elements
  for (unsigned int r = 0; r < masked.size(); r++) { 
    delete (CSCAddress*) masked.at(r);
  }
}

/**
 * @brief  Resets all detector map
 * @return 
 */
void CSCSummary::Reset() {
  CSCAddress adr;

  // Setting Zeros (no data) for each HW element (and beyond)
  adr.mask.side = adr.mask.station = adr.mask.layer = false;
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
 * @brief  Read Reporting Chamber histogram and fill in detector map.
 * @param  h2 Histogram to read
 * @return 
 */
void CSCSummary::ReadReportingChambers(TH2*& h2, const double threshold) {

  if(h2->GetXaxis()->GetXmin() <= 1 && h2->GetXaxis()->GetXmax() >= 36 &&
     h2->GetYaxis()->GetXmin() <= 1 && h2->GetYaxis()->GetXmax() >= 18) {

    CSCAddress adr;
    double z = 0.0;

    for(unsigned int x = 1; x <= 36; x++) {
      for(unsigned int y = 1; y <= 18; y++) {
        z = h2->GetBinContent(x, y);
        if(ChamberCoords(x, y, adr)) {
          if(z >= threshold) {
            SetValue(adr, 1);
          } else {
            SetValue(adr, 0);
          }
        }
      }
    }
  }
}

/**
 * @brief  Read Reporting Chamber histogram and fill in detector map based on
 * reference histogram.
 * @param  h2 Histogram to read
 * @param  refh2 Reference histogram of hit occupancies
 * @param  eps_min Minimum tolerance of difference (rate)
 * @param  Sfail Significance threshold for failure report
 * @return 
 */
void CSCSummary::ReadReportingChambersRef(TH2*& h2, TH2*& refh2, const double eps_min, const double Sfail) {

  if(h2->GetXaxis()->GetXmin() <= 1 && h2->GetXaxis()->GetXmax() >= 36 &&
     h2->GetYaxis()->GetXmin() <= 1 && h2->GetYaxis()->GetXmax() >= 18 &&
     refh2->GetXaxis()->GetXmin() <= 1 && refh2->GetXaxis()->GetXmax() >= 36 &&
     refh2->GetYaxis()->GetXmin() <= 1 && refh2->GetYaxis()->GetXmax() >= 18) {

    // Rate Factor calculation
    double num = 0.0, denum = 0.0;
    for(unsigned int x = 1; x <= 36; x++) {
      for(unsigned int y = 1; y <= 18; y++) {
        num += refh2->GetBinContent(x, y);
        if (h2->GetBinContent(x, y) > 0) {
          denum += (refh2->GetBinContent(x, y) * refh2->GetBinContent(x, y)) / h2->GetBinContent(x, y);
        }
      }
    }
    double factor = num / denum, eps_meas = 0.0;

    CSCAddress adr;
    unsigned int N = 0, n = 0;
    int val = 1;

    for(unsigned int x = 1; x <= 36; x++) {
      for(unsigned int y = 1; y <= 18; y++) {
        N = int(refh2->GetBinContent(x, y) * factor);
        n = int(h2->GetBinContent(x, y));
        if(ChamberCoords(x, y, adr)) {
          val = 1;
          if (n == 0) {
            val = 0;
          } else if (N > 0) {
            eps_meas = (1.0 * n) / (1.0 * N);
            if (eps_meas < eps_min) {
              if (SignificanceLevel(N, n, eps_min) > Sfail) {
                val = -1;
              }
              LOGINFO("ReadReportingChambersRef") << "eps_min = " << eps_min << 
                                                     ", Sfail = " << Sfail << 
                                                     ", eps_meas = " << eps_meas << 
                                                     ", N = " << N << 
                                                     ", n = " << n << 
                                                     ", S = " << SignificanceLevel(N, n, eps_min) << 
                                                     ", value = " << val;
            } else
            if (eps_meas > 1.0) {
              if (SignificanceLevelHot(N, n) > Sfail) {
                val = -1;
              }
            }
          }
          SetValue(adr, val);
        }
      }
    }

  }
}

/**
 * @brief  Read Error data for Chambers
 * @param  evs Histogram for number of events (total)
 * @param  err Histogram for number of errors
 * @param  eps_max Maximum tolerance of errors (rate)
 * @param  Sfail Significance threshold for failure report
 * @return 
 */
void CSCSummary::ReadErrorChambers(TH2*& evs, TH2*& err, const double eps_max, const double Sfail) {

  if(evs->GetXaxis()->GetXmin() <= 1 && evs->GetXaxis()->GetXmax() >= 36 &&
     evs->GetYaxis()->GetXmin() <= 1 && evs->GetYaxis()->GetXmax() >= 18 &&
     err->GetXaxis()->GetXmin() <= 1 && err->GetXaxis()->GetXmax() >= 36 &&
     err->GetYaxis()->GetXmin() <= 1 && err->GetYaxis()->GetXmax() >= 18) {

    CSCAddress adr;
    unsigned int N = 0, n = 0; 

    for(unsigned int x = 1; x <= 36; x++) {
      for(unsigned int y = 1; y <= 18; y++) {
        N = int(evs->GetBinContent(x, y));
        n = int(err->GetBinContent(x, y));
        if(ChamberCoords(x, y, adr)) {
          if(SignificanceLevel(N, n, eps_max) > Sfail) { 
            //LOGINFO("ReadErrorChambers") << " N = " << N << ", n = " << n << ", Salpha = " << SignificanceLevel(N, n, eps_max, true);
            SetValue(adr, -1);
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
void CSCSummary::Write(TH2*& h2, const unsigned int station) const {
  const CSCAddressBox* box;
  CSCAddress adr;

  if(station < 1 || station > N_STATIONS) return; 

  adr.mask.side = adr.mask.ring = adr.mask.chamber = adr.mask.layer = adr.mask.cfeb = adr.mask.hv = false;
  adr.mask.station = true;
  adr.station = station;

  unsigned int i = 0;

  while (detector.NextAddressBox(i, box, adr)) { 

    unsigned int x = 1 + (box->adr.side - 1) * 9 + (box->adr.ring - 1) * 3 + (box->adr.hv - 1);
    unsigned int y = 1 + (box->adr.chamber - 1) * 5 + (box->adr.cfeb - 1);
    h2->SetBinContent(x, y, GetValue(box->adr));

  }

  TString title = Form("ME%d Status: Physics Efficiency %.2f", station, GetEfficiencyArea(adr));
  h2->SetTitle(title);

}

/**
 * @brief  Write PhysicsReady Map to H2 histogram
 * @param  h2 Histogram to write map to
 * @return Fraction of active area 
 */
const float CSCSummary::WriteMap(TH2*& h2) const {

  const unsigned int NTICS = 100;
  unsigned int rep_el = 0, csc_el = 0;

  if(h2->GetXaxis()->GetXmin() <= 1 && h2->GetXaxis()->GetXmax() >= NTICS &&
     h2->GetYaxis()->GetXmin() <= 1 && h2->GetYaxis()->GetXmax() >= NTICS) {

    float xd = 5.0 / NTICS, yd = 1.0 * (2.0 * 3.14159) / NTICS;

    float xmin, xmax, ymin, ymax;

    for(unsigned int x = 0; x < NTICS; x++) {

      xmin = -2.5 + xd * x;
      xmax = xmin + xd;

      for(unsigned int y = 0; y < NTICS; y++) {

        h2->SetBinContent(x + 1, y + 1, 0);

        if (xmin == -2.5 || xmax == 2.5) continue;
        if (xmin >= -1 && xmax <= 1)     continue;

        ymin = yd * y;
        ymax = ymin + yd;

        switch(IsPhysicsReady(xmin, xmax, ymin, ymax)) {
          case -1:
            h2->SetBinContent(x + 1, y + 1, -1);
            break;
          case 0:
            rep_el++;
            break;
          case 1:
            h2->SetBinContent(x + 1, y + 1, 1);
            rep_el++;
            break;
          case 2:
            h2->SetBinContent(x + 1, y + 1, 2);
            rep_el++;
        }

        csc_el++;

      }
    }

  }

  return (csc_el == 0 ? 0.0 : (1.0 * rep_el) / csc_el);

}

/**
 * @brief  SetValue for the whole of detector
 * @param  value Value to set
 * @return 
 */
void CSCSummary::SetValue(const int value) {
  CSCAddress adr;
  adr.mask.side = adr.mask.station = adr.mask.ring = adr.mask.chamber = adr.mask.layer = adr.mask.cfeb = adr.mask.hv = false;
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

  if (!adr.mask.layer) {
    adr.mask.layer = true;
    for (adr.layer = 1; adr.layer <= N_LAYERS; adr.layer++) SetValue(adr, value);
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
      adr.layer > 0 && adr.layer <= N_LAYERS && adr.cfeb > 0 && adr.cfeb <= N_CFEBS && adr.hv > 0 && adr.hv <= N_HVS) {

    if (map[adr.side - 1][adr.station - 1][adr.ring - 1][adr.chamber - 1][adr.layer - 1][adr.cfeb - 1][adr.hv - 1] != 2 ){
      map[adr.side - 1][adr.station - 1][adr.ring - 1][adr.chamber - 1][adr.layer - 1][adr.cfeb - 1][adr.hv - 1] = value;
    }

  }

}

/**
 * @brief  Check if the current eta/phi polygon has at least 2 active HW
 * elements in the area
 * @param  xmin Eta min coordinate of the polygon
 * @param  xmax Eta max coordinate of the polygon
 * @param  ymin Phi min coordinate of the polygon
 * @param  ymax Phi max coordinate of the polygon
 * @return 1 if this polygon is ok for physics and reporting, 0 - if it is ok
 * but does not report, -1 - otherwise
 */
const int CSCSummary::IsPhysicsReady(const float xmin, const float xmax, const float ymin, const float ymax) const {

  float xpmin = (xmin < xmax ? xmin : xmax);
  float xpmax = (xmax > xmin ? xmax : xmin);
  float ypmin = (ymin < ymax ? ymin : ymax);
  float ypmax = (ymax > ymin ? ymax : ymin);

  if (xmin >= -1.0 && xmax <= 1.0) return 0; 

  CSCAddress adr;
  const CSCAddressBox *box;

  adr.mask.ring = adr.mask.chamber = adr.mask.layer = adr.mask.cfeb = adr.mask.hv = false;
  adr.mask.side = adr.mask.station = true;
  adr.side = (xmin > 0 ? 1 : 2);

  unsigned int rep_sum = 0;
  unsigned int nul_sum = 0;
  unsigned int err_sum = 0;
  unsigned int mas_sum = 0;

  for (adr.station = 1; adr.station <= N_STATIONS; adr.station++) {

    unsigned int i = 0;
    while(detector.NextAddressBox(i, box, adr)) {
      
      float xboxmin = (box->xmin < box->xmax ? box->xmin : box->xmax);
      float xboxmax = (box->xmax > box->xmin ? box->xmax : box->xmin);
      float yboxmin = (box->ymin < box->ymax ? box->ymin : box->ymax);
      float yboxmax = (box->ymax > box->ymin ? box->ymax : box->ymin);

      if ((xpmin < xboxmin && xpmax < xboxmin) || (xpmin > xboxmax && xpmax > xboxmax)) continue;
      if ((ypmin < yboxmin && ypmax < yboxmin) || (ypmin > yboxmax && ypmax > yboxmax)) continue;

      //std::cout << "Request: " << xmin << ", " << xmax << ", " << ymin << ", " << ymax << std::endl;
      //std::cout << "Respons: " << box->xmin << ", " << box->xmax << ", " << box->ymin << ", " << box->ymax << std::endl;
      //detector.PrintAddress(box->adr);

      switch(GetValue(box->adr)) {
        case -1:
          err_sum++;
          break;
        case 0:
          nul_sum++;
          break;
        case 1:
          rep_sum++;
          break;
        case 2:
          mas_sum++;
        }

    }

    if (rep_sum >= 2) return 1;

  }

  if (err_sum >= 1) return -1;
  if (mas_sum >= 1) return 2;

  return 0;

}

/**
 * @brief  Get efficiency of the whole detector
 * @param  
 * @return Detector efficiency rate (0..1)
 */
const double CSCSummary::GetEfficiencyHW() const {

  CSCAddress adr;
  adr.mask.side = adr.mask.station = adr.mask.ring = adr.mask.chamber = adr.mask.layer = adr.mask.cfeb = adr.mask.hv = false;
  return GetEfficiencyHW(adr);

}

/**
 * @brief  Get efficiency of the station
 * @param  
 * @return Detector efficiency rate (0..1)
 */
const double CSCSummary::GetEfficiencyHW(const unsigned int station) const {

  CSCAddress adr;
  adr.mask.side = adr.mask.station = adr.mask.ring = adr.mask.chamber = adr.mask.layer = adr.mask.cfeb = adr.mask.hv = false;

  if (station > 0 && station <= N_STATIONS) {
    adr.mask.station = true;
    adr.station = station;
  } else {
    return 0.0;
  }

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

  if (!adr.mask.layer) {
    adr.mask.layer = true;
    for (adr.layer = 1; adr.layer <= N_LAYERS; adr.layer++) sum += GetEfficiencyHW(adr);
    return sum / N_LAYERS;
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

  // if not error - then OK!
  if (GetValue(adr) >= 0) return 1.0;

  return 0.0;

}

/**
 * @brief  Get Efficiency area for the station
 * @param  station Station number 1..4
 * @return Reporting Area for the Station
 */
const double CSCSummary::GetEfficiencyArea(const unsigned int station) const {
  if (station <= 0 || station > N_STATIONS) return 0.0;

  CSCAddress adr;
  adr.mask.side = adr.mask.ring = adr.mask.chamber = adr.mask.layer = adr.mask.cfeb = adr.mask.hv = false;
  adr.station   = true;
  adr.station   = station;

  return GetEfficiencyArea(adr);
}

/**
 * @brief  Get Efficiency area for the address
 * @param  adr Address
 * @return Area in eta/phi space
 */
const double CSCSummary::GetEfficiencyArea(CSCAddress adr) const {
  double all_area = 1;

  if(adr.mask.side == adr.mask.ring == adr.mask.chamber == adr.mask.layer == adr.mask.cfeb == adr.mask.hv == false && adr.mask.station == true)
    all_area = detector.Area(adr.station);
  else
    all_area = detector.Area(adr);

  double rep_area = GetReportingArea(adr);
  return rep_area / all_area;
}

/**
 * @brief  Calculate the reporting area for the address
 * @param  adr Address to calculate
 * @return Area in eta/phi space
 */
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

  adr.mask.layer = false;
   
  // NOT errorous! 
  if (GetValue(adr) >= 0) {
    return detector.Area(adr);
  }

  return 0.0;

}
/**
 * @brief  Get value of some address (address must be fully filled except layers! otherwise function returns 0) 
 * If layer is ommited then function returns 1 if at least two layers for the HW element are live. 
 * If layer is ommited then function returns -1 if at least two layers for the HW element are errorous. 
 * If layer is ommited then function returns 2 if at least two layers for the HW element are off. 
 * @param  adr Address of atomic element to return value from
 * @return Value of the requested element
 */
const int CSCSummary::GetValue(const CSCAddress& adr) const {

  if ( adr.mask.side && adr.mask.station && adr.mask.ring && 
      adr.mask.chamber && adr.mask.cfeb && adr.mask.hv &&
      adr.side > 0 && adr.side <= N_SIDES && 
      adr.station > 0 && adr.station <= N_STATIONS && 
      adr.ring > 0 && adr.ring <= N_RINGS && 
      adr.chamber > 0 && adr.chamber <= N_CHAMBERS && 
      adr.cfeb > 0 && adr.cfeb <= N_CFEBS && 
      adr.hv > 0 && adr.hv <= N_HVS) {

    if (adr.mask.layer) {
      if (adr.layer > 0 && adr.layer <= N_LAYERS) {
        return map[adr.side - 1][adr.station - 1][adr.ring - 1][adr.chamber - 1][adr.layer - 1][adr.cfeb - 1][adr.hv - 1];
      }
    } else {
      unsigned int n_live_layers = 0;
      unsigned int n_err_layers  = 0;
      unsigned int n_off_layers  = 0;
      for (unsigned int layer = 1; layer < N_LAYERS; layer++) {
        int value = map[adr.side - 1][adr.station - 1][adr.ring - 1][adr.chamber - 1][layer - 1][adr.cfeb - 1][adr.hv - 1];
        switch (value) {
          case 1:
            n_live_layers ++;
            break;
          case -1:
            n_err_layers++;
            break;
          case 2:
            n_off_layers++;
            break;
        }
      }

      if (n_live_layers >= 2) return  1;
      if (n_err_layers  >= 2) return -1;
      if (n_off_layers  >= 2) return  2;

    }
  }

  return 0;

}

/**
 * @brief  Read HW element masks (strings), create Address and apply to detector map
 * @param  tokens Vector of mask strings
 * @return number of read and applied masks
 */
const unsigned int CSCSummary::setMaskedHWElements(std::vector<std::string>& tokens) {
  unsigned int applied = 0;
  for (unsigned int r = 0; r < tokens.size(); r++) {
    std::string token = (std::string) tokens.at(r);
    CSCAddress adr;
    if (detector.AddressFromString(token, adr)) {
      SetValue(adr, 2);
      applied++; 
    }
  }
  return applied;
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
  adr.mask.layer = adr.mask.cfeb = adr.mask.hv = false;

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
 * @brief  Calculate error significance alpha for the given number of errors
 * @param  N Number of events
 * @param  n Number of errors
 * @param  eps Rate of tolerance
 * @return Significance level
 */
const double CSCSummary::SignificanceLevel(const unsigned int N, const unsigned int n, const double eps) const {

  double l_eps = eps;
  if (l_eps <= 0.0) l_eps = 0.000001;
  if (l_eps >= 1.0) l_eps = 0.999999;

  double eps_meas = (1.0 * n) / (1.0 * N);
  double a = 1.0, b = 1.0;

  if (n > 0) {
    for (unsigned int r = 0; r < n; r++) a = a * (eps_meas / l_eps);
  }

  if (n > 0 && n < N) {
    for (unsigned int r = 0; r < (N - n); r++) b = b * (1 - eps_meas) / (1 - l_eps);
  }

  return sqrt(2.0 * log(a * b));

}

/**
 * @brief  Calculate error significance alpha for the given number of events
 * based on reference number of errors for "hot" elements: actual number of
 * events have to be larger then the reference.
 * @param  N number of reference events
 * @param  n number of actual events
 * @return error significance
 */
const double CSCSummary::SignificanceLevelHot(const unsigned int N, const unsigned int n) const {
  if (N > n) return 0.0;
  // no - n observed, ne - n expected
  double no = 1.0 * n, ne = 1.0 * N;
  return sqrt(2.0 * (no * (log(no / ne) - 1) + ne));
}

