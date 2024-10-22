/*
 * =====================================================================================
 *
 *       Filename:  Summary.cc
 *
 *    Description:  Class Summary implementation
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

#include "CSCDQM_Summary.h"

namespace cscdqm {

  /**
   * @brief  Constructor
   */
  Summary::Summary() : detector(NTICS, NTICS) { Reset(); }

  /**
   * @brief  Destructor
   */
  Summary::~Summary() {}

  /**
   * @brief  Resets all detector map
   */
  void Summary::Reset() {
    Address adr;
    bzero(&adr, sizeof(Address));

    /**  Setting Zeros (no data) for each HW element (and beyond) */
    adr.mask.side = adr.mask.station = adr.mask.layer = false;
    adr.mask.ring = adr.mask.chamber = adr.mask.cfeb = adr.mask.hv = true;
    for (adr.ring = 1; adr.ring <= N_RINGS; adr.ring++) {
      for (adr.chamber = 1; adr.chamber <= N_CHAMBERS; adr.chamber++) {
        for (adr.cfeb = 1; adr.cfeb <= N_CFEBS; adr.cfeb++) {
          for (adr.hv = 1; adr.hv <= N_HVS; adr.hv++) {
            for (unsigned int bit = 0; bit < HWSTATUSBITSETSIZE; bit++) {
              ReSetValue(adr, (HWStatusBit)bit);
            }
          }
        }
      }
    }
  }

  /**
   * @brief  Read Reporting Chamber histogram and fill in detector map.
   * @param  h2 Histogram to read
   * @param  threshold Min bin value to set HW element as reporting 
   */
  void Summary::ReadReportingChambers(const TH2*& h2, const double threshold) {
    if (h2->GetXaxis()->GetXmin() <= 1 && h2->GetXaxis()->GetXmax() >= 36 && h2->GetYaxis()->GetXmin() <= 1 &&
        h2->GetYaxis()->GetXmax() >= 18) {
      Address adr;
      bzero(&adr, sizeof(Address));
      double z = 0.0;

      for (unsigned int x = 1; x <= 36; x++) {
        for (unsigned int y = 1; y <= 18; y++) {
          z = h2->GetBinContent(x, y);
          if (ChamberCoordsToAddress(x, y, adr)) {
            if (z >= threshold) {
              SetValue(adr, DATA);
            } else {
              ReSetValue(adr, DATA);
            }
          }
        }
      }
    } else {
      LOG_WARN << "cscdqm::Summary.ReadReportingChambers routine. Wrong histogram dimensions!";
    }
  }

  /**
   * @brief  Read Reporting Chamber histogram and fill in detector map based on
   * reference histogram.
   * @param  h2 Histogram to read
   * @param  refh2 Reference histogram of hit occupancies
   * @param  cold_coef Minimum tolerance of difference (rate) to set COLD (not reporting) HW element
   * @param  cold_Sfail Significance threshold for COLD HW element
   * @param  hot_coef Minimum tolerance of difference (rate) to set HOT HW element
   * @param  hot_Sfail Significance threshold for HOT HW element
   */
  void Summary::ReadReportingChambersRef(const TH2*& h2,
                                         const TH2*& refh2,
                                         const double cold_coef,
                                         const double cold_Sfail,
                                         const double hot_coef,
                                         const double hot_Sfail) {
    if (h2->GetXaxis()->GetXmin() <= 1 && h2->GetXaxis()->GetXmax() >= 36 && h2->GetYaxis()->GetXmin() <= 1 &&
        h2->GetYaxis()->GetXmax() >= 18 && refh2->GetXaxis()->GetXmin() <= 1 && refh2->GetXaxis()->GetXmax() >= 36 &&
        refh2->GetYaxis()->GetXmin() <= 1 && refh2->GetYaxis()->GetXmax() >= 18) {
      /**  Rate Factor calculation */
      double num = 1.0, denum = 1.0;
      for (unsigned int x = 1; x <= 36; x++) {
        for (unsigned int y = 1; y <= 18; y++) {
          double Nij = h2->GetBinContent(x, y);
          double Nrefij = refh2->GetBinContent(x, y);
          if (Nij > 0) {
            num += Nrefij;
            denum += pow(Nrefij, 2.0) / Nij;
          }
        }
      }
      double factor = num / denum;

      Address adr;
      bzero(&adr, sizeof(Address));
      unsigned int N = 0, n = 0;

      for (unsigned int x = 1; x <= 36; x++) {
        for (unsigned int y = 1; y <= 18; y++) {
          N = int(refh2->GetBinContent(x, y) * factor);
          n = int(h2->GetBinContent(x, y));

          if (ChamberCoordsToAddress(x, y, adr)) {
            /**  Reset some bits */
            ReSetValue(adr, HOT);
            ReSetValue(adr, COLD);

            if (n == 0) {
              ReSetValue(adr, DATA);
            } else {
              SetValue(adr, DATA);
            }

            switch (Utility::checkOccupancy(N, n, cold_coef, hot_coef, cold_Sfail, hot_Sfail)) {
              case -1:
                SetValue(adr, COLD);

                /*
            std::cout << "adr = " << detector.AddressName(adr);
            std::cout << ", x = " << x << ", y = " << y;
            std::cout << ", value = " << GetValue(adr);
            std::cout << ", refh2 = " << refh2->GetBinContent(x, y);
            std::cout << ", factor = " << factor;
            std::cout << ", N = " << N;
            std::cout << ", n = " << n;
            std::cout << ", num = " << num;
            std::cout << ", denum = " << denum;
            std::cout << ", rate = " << (N > 0 ? n / N : 0);
            std::cout << ", cold_coef = " << cold_coef;
            std::cout << ", = COLD";
            std::cout << "\n";
              */

                break;
              case 1:
                SetValue(adr, HOT);

                /*
            std::cout << "adr = " << detector.AddressName(adr);
            std::cout << ", x = " << x << ", y = " << y;
            std::cout << ", value = " << GetValue(adr);
            std::cout << ", refh2 = " << refh2->GetBinContent(x, y);
            std::cout << ", factor = " << factor;
            std::cout << ", N = " << N;
            std::cout << ", n = " << n;
            std::cout << ", num = " << num;
            std::cout << ", denum = " << denum;
            std::cout << ", rate = " << (N > 0 ? n / N : 0);
            std::cout << ", hot_coef = " << hot_coef;
            std::cout << ", = HOT";
            std::cout << "\n";
              */

                break;
            };
          }
        }
      }

    } else {
      LOG_WARN << "cscdqm::Summary.ReadReportingChambersRef routine. Wrong histogram dimensions!";
    }
  }

  /**
   * @brief  Read Error data for Chambers
   * @param  evs Histogram for number of events (total)
   * @param  err Histogram for number of errors
   * @param  bit Error bit to set
   * @param  eps_max Maximum tolerance of errors (rate)
   * @param  Sfail Significance threshold for failure report
   */
  void Summary::ReadErrorChambers(
      const TH2*& evs, const TH2*& err, const HWStatusBit bit, const double eps_max, const double Sfail) {
    if (evs->GetXaxis()->GetXmin() <= 1 && evs->GetXaxis()->GetXmax() >= 36 && evs->GetYaxis()->GetXmin() <= 1 &&
        evs->GetYaxis()->GetXmax() >= 18 && err->GetXaxis()->GetXmin() <= 1 && err->GetXaxis()->GetXmax() >= 36 &&
        err->GetYaxis()->GetXmin() <= 1 && err->GetYaxis()->GetXmax() >= 18) {
      Address adr;
      bzero(&adr, sizeof(Address));
      unsigned int N = 0, n = 0;

      for (unsigned int x = 1; x <= 36; x++) {
        for (unsigned int y = 1; y <= 18; y++) {
          N = int(evs->GetBinContent(x, y));
          n = int(err->GetBinContent(x, y));
          if (ChamberCoordsToAddress(x, y, adr)) {
            if (Utility::checkError(N, n, eps_max, Sfail)) {
              SetValue(adr, bit);
            } else {
              ReSetValue(adr, bit);
            }
          }
        }
      }
    } else {
      LOG_WARN << "cscdqm::Summary.ReadErrorChambers routine. Wrong histogram dimensions!";
    }
  }

  /**
   * @brief  Write detector map to H1 histogram (linear data) for the selected adr.station
   * @param  h2 Histogram to write data to
   * @param  station station number (1-4) to write data for
   */
  void Summary::Write(TH2*& h2, const unsigned int station) const {
    const AddressBox* box;
    Address adr, tadr;
    bzero(&adr, sizeof(Address));
    bzero(&tadr, sizeof(Address));
    float area_all = 0.0, area_rep = 0.0;

    if (station < 1 || station > N_STATIONS)
      return;

    adr.mask.side = adr.mask.ring = adr.mask.chamber = adr.mask.layer = adr.mask.cfeb = adr.mask.hv = false;
    adr.mask.station = true;
    adr.station = station;

    unsigned int i = 0;

    while (detector.NextAddressBox(i, box, adr)) {
      unsigned int x = 1 + (box->adr.side - 1) * 9 + (box->adr.ring - 1) * 3 + (box->adr.hv - 1);
      unsigned int y = 1 + (box->adr.chamber - 1) * 5 + (box->adr.cfeb - 1);

      tadr = box->adr;
      HWStatusBitSet status = GetValue(tadr);

      float area_box = fabs((box->xmax - box->xmin) * (box->ymax - box->ymin));

      if (status.test(MASKED)) {
        h2->SetBinContent(x, y, 2.0);
      } else {
        area_all += area_box;
        if (HWSTATUSANYERROR(status)) {
          h2->SetBinContent(x, y, -1.0);
        } else {
          area_rep += area_box;
          if (status.test(DATA)) {
            h2->SetBinContent(x, y, 1.0);
          } else {
            h2->SetBinContent(x, y, 0.0);
          }
        }
      }
    }

    TString title = Form("ME%d Status: Physics Efficiency %.2f%%", station, (area_rep / area_all) * 100.0);
    h2->SetTitle(title);
  }

  /**
   * @brief  Write PhysicsReady Map to H2 histogram
   * @param  h2 Histogram to write map to
   */
  void Summary::WriteMap(TH2*& h2) {
    unsigned int rep_el = 0, csc_el = 0;

    if (h2->GetXaxis()->GetXmin() <= 1 && h2->GetXaxis()->GetXmax() >= NTICS && h2->GetYaxis()->GetXmin() <= 1 &&
        h2->GetYaxis()->GetXmax() >= NTICS) {
      float xd = 5.0 / NTICS;

      float xmin, xmax;

      for (unsigned int x = 0; x < NTICS; x++) {
        xmin = -2.5 + xd * x;
        xmax = xmin + xd;

        for (unsigned int y = 0; y < NTICS; y++) {
          double value = 0.0;

          if (xmin == -2.5 || xmax == 2.5)
            continue;
          if (xmin >= -1 && xmax <= 1)
            continue;

          switch (IsPhysicsReady(x, y)) {
            case -1:
              value = -1.0;
              break;
            case 0:
              value = 0.0;
              rep_el++;
              break;
            case 1:
              value = 1.0;
              rep_el++;
              break;
            case 2:
              value = 2.0;
              rep_el++;
          }

          h2->SetBinContent(x + 1, y + 1, value);
          csc_el++;
        }
      }

    } else {
      LOG_WARN << "cscdqm::Summary.WriteMap routine. Wrong histogram dimensions!";
    }

    TString title =
        Form("EMU Status: Physics Efficiency %.2f%%", (csc_el == 0 ? 0.0 : (1.0 * rep_el) / csc_el) * 100.0);
    h2->SetTitle(title);
  }

  /**
   * @brief  Write State information to chamber histogram
   * @param  h2 histogram to write to
   * @param  mask mask of errors to check while writing
   * @param  value to write to if state fits mask
   * @param  reset should all chamber states be reseted to 0 prior writing?
   * @param  op_any Should chamber be marked as errorous on any bit in mask? false - for all.
   */
  void Summary::WriteChamberState(TH2*& h2, const int mask, const int value, const bool reset, const bool op_any) const {
    if (h2->GetXaxis()->GetXmin() <= 1 && h2->GetXaxis()->GetXmax() >= 36 && h2->GetYaxis()->GetXmin() <= 1 &&
        h2->GetYaxis()->GetXmax() >= 18) {
      unsigned int x, y;
      Address adr;
      bzero(&adr, sizeof(Address));

      adr.mask.side = adr.mask.station = adr.mask.ring = adr.mask.chamber = true;
      adr.mask.layer = adr.mask.cfeb = adr.mask.hv = false;

      for (adr.side = 1; adr.side <= N_SIDES; adr.side++) {
        for (adr.station = 1; adr.station <= N_STATIONS; adr.station++) {
          for (adr.ring = 1; adr.ring <= detector.NumberOfRings(adr.station); adr.ring++) {
            for (adr.chamber = 1; adr.chamber <= detector.NumberOfChambers(adr.station, adr.ring); adr.chamber++) {
              if (ChamberAddressToCoords(adr, x, y)) {
                HWStatusBitSet hwValue = GetValue(adr);
                bool hit = (op_any ? HWSTATUSANY(hwValue, mask) : HWSTATUSEQUALS(hwValue, mask));

                // std::cout << "x = " << x << ", y = " << y << ", value = " << GetValue(adr) << std::endl;
                // std::cout << "adr = " << detector.AddressName(adr) << ", x = " << x << ", y = " << y << ", value = " << GetValue(adr) << std::endl;
                if (hit) {
                  h2->SetBinContent(x, y, 1.0 * value);
                } else if (reset) {
                  h2->SetBinContent(x, y, 0.0);
                }
              }
            }
          }
        }
      }

    } else {
      LOG_WARN << "cscdqm::Summary.WriteChamberState routine. Wrong histogram dimensions!";
    }
  }

  /**
   * @brief  ReSetValue for the whole of detector
   * @param  bit Status bit to set
   */
  void Summary::ReSetValue(const HWStatusBit bit) { SetValue(bit, 0); }

  /**
   * @brief  ReSet value recursivelly by following the supplied address
   * @param  adr Address to be updated
   * @param  bit Status bit to set
   */
  void Summary::ReSetValue(const Address& adr, const HWStatusBit bit) { SetValue(adr, bit, 0); }

  /**
   * @brief  SetValue for the whole of detector
   * @param  bit Status bit to set
   * @param  value Value to set
   */
  void Summary::SetValue(const HWStatusBit bit, const int value) {
    Address adr;
    bzero(&adr, sizeof(Address));
    adr.mask.side = adr.mask.station = adr.mask.ring = adr.mask.chamber = adr.mask.layer = adr.mask.cfeb = adr.mask.hv =
        false;
    SetValue(adr, bit, value);
  }

  /**
   * @brief  Set value recursivelly by following the supplied address
   * @param  adr Address to be updated
   * @param  bit Status bit to set
   * @param  value Value to be set
   */
  void Summary::SetValue(Address adr, const HWStatusBit bit, const int value) {
    if (!adr.mask.side) {
      adr.mask.side = true;
      for (adr.side = 1; adr.side <= N_SIDES; adr.side++)
        SetValue(adr, bit, value);
      return;
    }

    if (!adr.mask.station) {
      adr.mask.station = true;
      for (adr.station = 1; adr.station <= N_STATIONS; adr.station++)
        SetValue(adr, bit, value);
      return;
    }

    if (!adr.mask.ring) {
      adr.mask.ring = true;
      for (adr.ring = 1; adr.ring <= detector.NumberOfRings(adr.station); adr.ring++)
        SetValue(adr, bit, value);
      return;
    }

    if (!adr.mask.chamber) {
      adr.mask.chamber = true;
      for (adr.chamber = 1; adr.chamber <= detector.NumberOfChambers(adr.station, adr.ring); adr.chamber++)
        SetValue(adr, bit, value);
      return;
    }

    if (!adr.mask.layer) {
      adr.mask.layer = true;
      for (adr.layer = 1; adr.layer <= N_LAYERS; adr.layer++)
        SetValue(adr, bit, value);
      return;
    }

    if (!adr.mask.cfeb) {
      adr.mask.cfeb = true;
      for (adr.cfeb = 1; adr.cfeb <= detector.NumberOfChamberCFEBs(adr.station, adr.ring); adr.cfeb++)
        SetValue(adr, bit, value);
      return;
    }

    if (!adr.mask.hv) {
      adr.mask.hv = true;
      for (adr.hv = 1; adr.hv <= detector.NumberOfChamberHVs(adr.station, adr.ring); adr.hv++)
        SetValue(adr, bit, value);
      return;
    }

    if (adr.side > 0 && adr.side <= N_SIDES && adr.station > 0 && adr.station <= N_STATIONS && adr.ring > 0 &&
        adr.ring <= N_RINGS && adr.chamber > 0 && adr.chamber <= N_CHAMBERS && adr.layer > 0 && adr.layer <= N_LAYERS &&
        adr.cfeb > 0 && adr.cfeb <= N_CFEBS && adr.hv > 0 && adr.hv <= N_HVS) {
      map[adr.side - 1][adr.station - 1][adr.ring - 1][adr.chamber - 1][adr.layer - 1][adr.cfeb - 1][adr.hv - 1].set(
          bit, value);
    }
  }

  /**
   * @brief  Check if the current partition element (aka eta/phi polygon) has at least 2 active HW
   * elements in the area
   * @param  px partition element index in x axis
   * @param  py partition element index in y axis
   * @return 1 if this polygon is ok for physics and reporting, 0 - if it is ok
   * but does not report, -1 - otherwise
   */
  const int Summary::IsPhysicsReady(const unsigned int px, const unsigned int py) {
    AddressBox* box;

    HWStatusBitSet status[N_STATIONS];

    unsigned int i = 0;
    while (detector.NextAddressBoxByPartition(i, px, py, box)) {
      status[box->adr.station - 1] |= GetValue(box->adr);
    }

    unsigned int cdata = 0, cerror = 0, cmask = 0;
    for (unsigned int i = 0; i < N_STATIONS; i++) {
      if (HWSTATUSANYERROR(status[i])) {
        cerror++;
      } else {
        if (status[i].test(MASKED))
          cmask++;
        if (status[i].test(DATA))
          cdata++;
      }
    }

    /**  If at least 2 stations with data and without errors = OK */
    if (cdata > 1)
      return 1;
    /**  Else, if at least one station errorous = ERROR */
    if (cerror > 0)
      return -1;
    /**  Else, if at least one station masked = MASKED */
    if (cmask > 0)
      return 2;
    /**  Else, not sufficient data = OK */
    return 0;
  }

  /**
   * @brief  Get efficiency of the whole detector
   * @return Detector efficiency rate (0..1)
   */
  const double Summary::GetEfficiencyHW() const {
    Address adr;
    bzero(&adr, sizeof(Address));
    adr.mask.side = adr.mask.station = adr.mask.ring = adr.mask.chamber = adr.mask.layer = adr.mask.cfeb = adr.mask.hv =
        false;
    return GetEfficiencyHW(adr);
  }

  /**
   * @brief  Get efficiency of the station
   * @param  station Station number
   * @return Detector efficiency rate (0..1)
   */
  const double Summary::GetEfficiencyHW(const unsigned int station) const {
    Address adr;
    bzero(&adr, sizeof(Address));
    adr.mask.side = adr.mask.station = adr.mask.ring = adr.mask.chamber = adr.mask.layer = adr.mask.cfeb = adr.mask.hv =
        false;

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
  const double Summary::GetEfficiencyHW(Address adr) const {
    double sum = 0.0;
    if (!adr.mask.side) {
      adr.mask.side = true;
      for (adr.side = 1; adr.side <= N_SIDES; adr.side++)
        sum += GetEfficiencyHW(adr);
      return sum / N_SIDES;
    }

    if (!adr.mask.station) {
      adr.mask.station = true;
      for (adr.station = 1; adr.station <= N_STATIONS; adr.station++)
        sum += GetEfficiencyHW(adr);
      return sum / N_STATIONS;
    }

    if (!adr.mask.ring) {
      adr.mask.ring = true;
      for (adr.ring = 1; adr.ring <= detector.NumberOfRings(adr.station); adr.ring++)
        sum += GetEfficiencyHW(adr);
      return sum / detector.NumberOfRings(adr.station);
    }

    if (!adr.mask.chamber) {
      adr.mask.chamber = true;
      for (adr.chamber = 1; adr.chamber <= detector.NumberOfChambers(adr.station, adr.ring); adr.chamber++)
        sum += GetEfficiencyHW(adr);
      return sum / detector.NumberOfChambers(adr.station, adr.ring);
    }

    if (!adr.mask.layer) {
      adr.mask.layer = true;
      for (adr.layer = 1; adr.layer <= N_LAYERS; adr.layer++)
        sum += GetEfficiencyHW(adr);
      return sum / N_LAYERS;
    }

    if (!adr.mask.cfeb) {
      adr.mask.cfeb = true;
      for (adr.cfeb = 1; adr.cfeb <= detector.NumberOfChamberCFEBs(adr.station, adr.ring); adr.cfeb++)
        sum += GetEfficiencyHW(adr);
      return sum / detector.NumberOfChamberCFEBs(adr.station, adr.ring);
    }

    if (!adr.mask.hv) {
      adr.mask.hv = true;
      for (adr.hv = 1; adr.hv <= detector.NumberOfChamberHVs(adr.station, adr.ring); adr.hv++)
        sum += GetEfficiencyHW(adr);
      return sum / detector.NumberOfChamberHVs(adr.station, adr.ring);
    }

    /**  if not error - then OK! */
    HWStatusBitSet status = GetValue(adr);
    if (HWSTATUSANYERROR(status))
      return 0.0;
    return 1.0;
  }

  /**
   * @brief  Get Efficiency area for the station
   * @param  station Station number 1..4
   * @return Reporting Area for the Station
   */
  const double Summary::GetEfficiencyArea(const unsigned int station) const {
    if (station <= 0 || station > N_STATIONS)
      return 0.0;

    Address adr;
    bzero(&adr, sizeof(Address));
    adr.mask.side = adr.mask.ring = adr.mask.chamber = adr.mask.layer = adr.mask.cfeb = adr.mask.hv = false;
    adr.mask.station = true;
    adr.station = station;

    return GetEfficiencyArea(adr);
  }

  /**
   * @brief  Get Efficiency area for the address
   * @param  adr Address
   * @return Area in eta/phi space
   */
  const double Summary::GetEfficiencyArea(const Address& adr) const {
    double all_area = 1;

    if ((adr.mask.side == false) && (adr.mask.ring == false) && (adr.mask.chamber == false) &&
        (adr.mask.layer == false) && (adr.mask.cfeb == false) && (adr.mask.hv == false) && (adr.mask.station == true))
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
  const double Summary::GetReportingArea(Address adr) const {
    double sum = 0.0;
    if (!adr.mask.side) {
      adr.mask.side = true;
      for (adr.side = 1; adr.side <= N_SIDES; adr.side++)
        sum += GetReportingArea(adr);
      return sum;
    }

    if (!adr.mask.station) {
      adr.mask.station = true;
      for (adr.station = 1; adr.station <= N_STATIONS; adr.station++)
        sum += GetReportingArea(adr);
      return sum;
    }

    if (!adr.mask.ring) {
      adr.mask.ring = true;
      for (adr.ring = 1; adr.ring <= detector.NumberOfRings(adr.station); adr.ring++)
        sum += GetReportingArea(adr);
      return sum;
    }

    if (!adr.mask.chamber) {
      adr.mask.chamber = true;
      for (adr.chamber = 1; adr.chamber <= detector.NumberOfChambers(adr.station, adr.ring); adr.chamber++)
        sum += GetReportingArea(adr);
      return sum;
    }

    if (!adr.mask.cfeb) {
      adr.mask.cfeb = true;
      for (adr.cfeb = 1; adr.cfeb <= detector.NumberOfChamberCFEBs(adr.station, adr.ring); adr.cfeb++)
        sum += GetReportingArea(adr);
      return sum;
    }

    if (!adr.mask.hv) {
      adr.mask.hv = true;
      for (adr.hv = 1; adr.hv <= detector.NumberOfChamberHVs(adr.station, adr.ring); adr.hv++)
        sum += GetReportingArea(adr);
      return sum;
    }

    adr.mask.layer = false;

    /**  NOT errorous!  */
    HWStatusBitSet status = GetValue(adr);
    if (!HWSTATUSANYERROR(status)) {
      return detector.Area(adr);
    }
    return 0.0;
  }

  /**
   * @brief  Check if chamber is in standby?
   * @param  side Side
   * @param  station Station
   * @param  ring Ring
   * @param  chamber Chamber
   * @return true if chamber is in standby, false - otherwise
   */
  bool Summary::isChamberStandby(unsigned int side,
                                 unsigned int station,
                                 unsigned int ring,
                                 unsigned int chamber) const {
    Address adr;
    bzero(&adr, sizeof(Address));
    adr.mask.side = adr.mask.station = adr.mask.ring = adr.mask.chamber = true;
    adr.mask.layer = adr.mask.cfeb = adr.mask.hv = false;
    adr.side = side;
    adr.station = station;
    adr.ring = ring;
    adr.chamber = chamber;

    //std::cout << adr << " = " << HWSTATUSANY(GetValue(adr), 0x1000) << "\n";

    return HWSTATUSANY(GetValue(adr), 0x1000);
  }

  /**
   * @brief  Check if chamber is in standby?
   * @param  cid Chamber identifier
   * @return true if chamber is in standby, false - otherwise
   */
  bool Summary::isChamberStandby(CSCDetId cid) const {
    return isChamberStandby(cid.endcap(), cid.station(), cid.ring(), cid.chamber());
  }

  /**
   * @brief  Get value of some address 
   * @param  adr Address of atomic element to return value from
   * @return Value of the requested element
   */
  const HWStatusBitSet Summary::GetValue(Address adr) const {
    HWStatusBitSet state;
    state.reset();

    if (!adr.mask.side) {
      adr.mask.side = true;
      for (adr.side = 1; adr.side <= N_SIDES; adr.side++)
        state |= GetValue(adr);
      return state;
    }

    if (!adr.mask.station) {
      adr.mask.station = true;
      for (adr.station = 1; adr.station <= N_STATIONS; adr.station++)
        state |= GetValue(adr);
      return state;
    }

    if (!adr.mask.ring) {
      adr.mask.ring = true;
      for (adr.ring = 1; adr.ring <= detector.NumberOfRings(adr.station); adr.ring++)
        state |= GetValue(adr);
      return state;
    }

    if (!adr.mask.chamber) {
      adr.mask.chamber = true;
      for (adr.chamber = 1; adr.chamber <= detector.NumberOfChambers(adr.station, adr.ring); adr.chamber++)
        state |= GetValue(adr);
      return state;
    }

    if (!adr.mask.layer) {
      adr.mask.layer = true;
      for (adr.layer = 1; adr.layer <= N_LAYERS; adr.layer++)
        state |= GetValue(adr);
      return state;
    }

    if (!adr.mask.cfeb) {
      adr.mask.cfeb = true;
      for (adr.cfeb = 1; adr.cfeb <= detector.NumberOfChamberCFEBs(adr.station, adr.ring); adr.cfeb++)
        state |= GetValue(adr);
      return state;
    }

    if (!adr.mask.hv) {
      adr.mask.hv = true;
      for (adr.hv = 1; adr.hv <= detector.NumberOfChamberHVs(adr.station, adr.ring); adr.hv++)
        state |= GetValue(adr);
      return state;
    }

    return map[adr.side - 1][adr.station - 1][adr.ring - 1][adr.chamber - 1][adr.layer - 1][adr.cfeb - 1][adr.hv - 1];
  }

  /**
   * @brief  Read HW element masks (strings), create Address and apply to detector map
   * @param  tokens Vector of mask strings
   * @return number of read and applied masks
   */
  const unsigned int Summary::setMaskedHWElements(std::vector<std::string>& tokens) {
    unsigned int applied = 0;

    for (unsigned int r = 0; r < tokens.size(); r++) {
      std::string token = (std::string)tokens.at(r);
      Address adr;
      if (detector.AddressFromString(token, adr)) {
        SetValue(adr, MASKED);
        applied++;
      }
    }
    return applied;
  }

  /**
   * @brief  Calculate Address from CSCChamberMap histogram coordinates 
   * @param  x X coordinate of histogram
   * @param  y Y coordinate of histogram
   * @param  adr Address to be filled in and returned
   * @return true if address was found and filled, false - otherwise
   */
  const bool Summary::ChamberCoordsToAddress(const unsigned int x, const unsigned int y, Address& adr) const {
    if (x < 1 || x > 36 || y < 1 || y > 18)
      return false;

    adr.mask.side = adr.mask.station = adr.mask.ring = adr.mask.chamber = true;
    adr.mask.layer = adr.mask.cfeb = adr.mask.hv = false;

    if (y < 10)
      adr.side = 2;
    else
      adr.side = 1;

    adr.chamber = x;

    if (y == 1 || y == 18) {
      adr.station = 4;
      adr.ring = 2;
    } else if (y == 2 || y == 17) {
      adr.station = 4;
      adr.ring = 1;
    } else if (y == 3 || y == 16) {
      adr.station = 3;
      adr.ring = 2;
    } else if (y == 4 || y == 15) {
      adr.station = 3;
      adr.ring = 1;
    } else if (y == 5 || y == 14) {
      adr.station = 2;
      adr.ring = 2;
    } else if (y == 6 || y == 13) {
      adr.station = 2;
      adr.ring = 1;
    } else if (y == 7 || y == 12) {
      adr.station = 1;
      adr.ring = 3;
    } else if (y == 8 || y == 11) {
      adr.station = 1;
      adr.ring = 2;
    } else if (y == 9 || y == 10) {
      adr.station = 1;
      adr.ring = 1;
    }

    return true;
  }

  /**
   * @brief  Calculate CSCChamberMap histogram coordinates from Address
   * @param  adr Address
   * @param  x X coordinate of histogram to be returned
   * @param  y Y coordinate of histogram to be returned
   * @return true if coords filled, false - otherwise
   */
  const bool Summary::ChamberAddressToCoords(const Address& adr, unsigned int& x, unsigned int& y) const {
    if (!adr.mask.side || !adr.mask.station || !adr.mask.ring || !adr.mask.chamber)
      return false;

    x = adr.chamber;
    y = 0;

    if (adr.side == 1) {
      switch (adr.station) {
        case 1:
          y = 10;
          if (adr.ring == 2)
            y = 11;
          if (adr.ring == 3)
            y = 12;
          break;
        case 2:
          y = 13;
          if (adr.ring == 2)
            y = 14;
          break;
        case 3:
          y = 15;
          if (adr.ring == 2)
            y = 16;
          break;
        case 4:
          y = 17;
          if (adr.ring == 2)
            y = 18;
          break;
      }
    } else if (adr.side == 2) {
      switch (adr.station) {
        case 1:
          y = 7;
          if (adr.ring == 2)
            y = 8;
          if (adr.ring == 1)
            y = 9;
          break;
        case 2:
          y = 5;
          if (adr.ring == 1)
            y = 6;
          break;
        case 3:
          y = 3;
          if (adr.ring == 1)
            y = 4;
          break;
        case 4:
          y = 1;
          if (adr.ring == 1)
            y = 2;
          break;
      }
    }

    return true;
  }

}  // namespace cscdqm
