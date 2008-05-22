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

#define N_SIDES    2
#define N_STATIONS 4
#define N_RINGS    3
#define N_CHAMBERS 36
#define N_CFEBS    5
#define N_HVS      5

struct CSCAddressMask {
  bool side;
  bool station;
  bool ring;
  bool chamber;
  bool cfeb;
  bool hv;
};

struct CSCAddress {
  unsigned int side;
  unsigned int station;
  unsigned int ring;
  unsigned int chamber;
  unsigned int cfeb;
  unsigned int hv;
  CSCAddressMask mask;
};

class CSCSummary {

  public:

    CSCSummary();

    void Reset();

    void Read(TH1*& h1);
    void ReadChambers(TH2*& h2, const double threshold = 1);

    void Write(TH1*& h1);
    void Write(TH1*& h1, const unsigned int station);

    void SetValue(const int value);
    void SetValue(CSCAddress adr, const int value);
    const int GetValue(CSCAddress adr);

    const double GetEfficiency();
    const double GetEfficiency(CSCAddress adr); 

    void PrintAddress(const CSCAddress& adr);

    const unsigned int NumberOfRings(const unsigned int station);
    const unsigned int NumberOfChambers(const unsigned int station, const unsigned int ring);
    const unsigned int NumberOfChamberCFEBs(const unsigned int station, const unsigned int ring);
    const unsigned int NumberOfChamberHVs(const unsigned int station, const unsigned int ring);
    const bool ChamberCoords(const unsigned int x, const unsigned int y, CSCAddress& adr);

  private:

    int map[N_SIDES][N_STATIONS][N_RINGS][N_CHAMBERS][N_CFEBS][N_HVS];

};
