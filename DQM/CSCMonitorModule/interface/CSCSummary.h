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
#include "DQM/CSCMonitorModule/interface/CSCDetector.h"

class CSCSummary {

  public:

    CSCSummary();
    ~CSCSummary();

    void Reset();

    const CSCDetector Detector() const { return detector; }

    void ReadReportingChambers(TH2*& h2, const double threshold = 1.0);
    void ReadReportingChambersRef(TH2*& h2, TH2*& refh2, const double eps_min = 0.1, const double Sfail = 5.0);
    void ReadErrorChambers(TH2*& evs, TH2*& err, const double eps_max = 0.1, const double Sfail = 5.0);

    const unsigned int setMaskedHWElements(std::vector<std::string>& tokens);

    void Write(TH2*& h2, const unsigned int station) const;
    const float WriteMap(TH2*& h2) const;

    void SetValue(const int value);
    void SetValue(CSCAddress adr, const int value);

    const int GetValue(const CSCAddress& adr) const;
    const bool IsPhysicsReady(const float xmin, const float xmax, const float ymin, const float ymax) const;

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
    // Possible values and appropriate color codes in maps:
    // - null  = not covered (18, grey)
    // -  0    = no data (10, white)
    // - +1    = reporting with data, OK (8, green)
    // - +2    = not in readout, i.e. swithed off by mask (15, dark grey)
    // - -1    = error (2, red) 
    // - other = application error (6, magenta)
    int map[N_SIDES][N_STATIONS][N_RINGS][N_CHAMBERS][N_LAYERS][N_CFEBS][N_HVS];

    std::vector<CSCAddress*> masked;

    CSCDetector detector;

};

#endif
