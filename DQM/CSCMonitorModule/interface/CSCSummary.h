/*
 * =====================================================================================
 *
 *       Filename:  CSCSummary.h
 *
 *    Description:  CSC summary map and appropriate functions.
 *
 *        Version:  1.0
 *        Created:  05/19/2008 10:52:21 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), Valdas.Rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 * =====================================================================================
 */

#ifndef CSCSummary_H
#define CSCSummary_H

#include <TH2.h>
#include <math.h>
#include <vector>
#include <bitset>
#include "DQM/CSCMonitorModule/interface/CSCDetector.h"

#define HWSTATUSBITSETSIZE    11
#define ANYERROR(s)           (\
                                s.test(HOT) || \
                                s.test(COLD) || \
                                s.test(FORMAT_ERR) || \
                                s.test(L1SYNC_ERR) || \
                                s.test(FIFOFULL_ERR) || \
                                s.test(INPUTTO_ERR) || \
                                s.test(NODATA_ALCT) || \
                                s.test(NODATA_CLCT) || \
                                s.test(NODATA_CFEB)\
                              )

enum HWStatusBit {
  DATA,         // Data available (reporting)
  MASKED,       // HW element was masked out (not in readout)
  HOT,          // HW element is hot by comparing with reference histogram 
  COLD,         // HW element is cold comparing with reference histogram
  FORMAT_ERR,   // Format errors
  L1SYNC_ERR,   // L1A out of sync errors
  FIFOFULL_ERR, // DMB FIFO full error
  INPUTTO_ERR,  // DMB Input timeout error
  NODATA_ALCT,  // No ALCT data
  NODATA_CLCT,  // No CLCT data
  NODATA_CFEB   // No CFEB data
};

typedef std::bitset<HWSTATUSBITSETSIZE> HWStatusBitSet;

class CSCSummary {

  public:

    CSCSummary();
    ~CSCSummary();

    void Reset();

    const CSCDetector Detector() const { return detector; }

    void ReadReportingChambers(TH2*& h2, const double threshold = 1.0);
    void ReadReportingChambersRef(TH2*& h2, TH2*& refh2, const double cold_coef = 0.1, const double cold_Sfail = 5.0, const double hot_coef = 2.0, const double hot_Sfail = 5.0);
    void ReadErrorChambers(TH2*& evs, TH2*& err, const HWStatusBit bit, const double eps_max = 0.1, const double Sfail = 5.0);

    const unsigned int setMaskedHWElements(std::vector<std::string>& tokens);

    void Write(TH2*& h2, const unsigned int station) const;
    const float WriteMap(TH2*& h2) const;

    void ReSetValue(const HWStatusBit bit);
    void ReSetValue(CSCAddress adr, const HWStatusBit bit);
    void SetValue(const HWStatusBit bit, const int value = 1);
    void SetValue(CSCAddress adr, const HWStatusBit bit, const int value = 1);

    const HWStatusBitSet GetValue(const CSCAddress& adr) const;
    const unsigned long IsPhysicsReady(const float xmin, const float xmax, const float ymin, const float ymax) const;

    const double GetEfficiencyHW() const;
    const double GetEfficiencyHW(const unsigned int station) const;
    const double GetEfficiencyHW(CSCAddress adr) const; 
    const double GetEfficiencyArea(const unsigned int station) const; 
    const double GetEfficiencyArea(CSCAddress adr) const; 

  private:

    const bool ChamberCoords(const unsigned int x, const unsigned int y, CSCAddress& adr) const;
    const double GetReportingArea(CSCAddress adr) const; 
    const double SignificanceLevel(const unsigned int N, const unsigned int n, const double eps) const;
    const double SignificanceLevelHot(const unsigned int N, const unsigned int n) const;

    // Atomic HW element status matrix
    HWStatusBitSet map[N_SIDES][N_STATIONS][N_RINGS][N_CHAMBERS][N_LAYERS][N_CFEBS][N_HVS];

    std::vector<CSCAddress*> masked;

    CSCDetector detector;

};

#endif
