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

#include <TH2.h>
#include <math.h>
#include "DQM/CSCMonitorModule/interface/CSCDetector.h"

class CSCSummary {

  public:

    CSCSummary();

    void Reset();

    const CSCDetector Detector() const { return detector; }

    void Read(TH1*& h1);
    void ReadChambers(TH2*& h2, const double threshold = 1);

    void Write(TH1*& h1);
    void Write(TH1*& h1, const unsigned int station);

    void SetValue(const int value);
    void SetValue(CSCAddress adr, const int value);

    const int GetValue(const CSCAddress& adr) const;
    const bool IsPhysicsReady(const float xmin, const float xmax, const float ymin, const float ymax) const;

    const double GetEfficiencyHW() const;
    const double GetEfficiencyHW(CSCAddress adr) const; 
    const double GetEfficiencyArea(CSCAddress adr) const; 

  private:

    const bool ChamberCoords(const unsigned int x, const unsigned int y, CSCAddress& adr) const;
    const double GetReportingArea(CSCAddress adr) const; 

    int map[N_SIDES][N_STATIONS][N_RINGS][N_CHAMBERS][N_CFEBS][N_HVS];
    CSCDetector detector;

};
