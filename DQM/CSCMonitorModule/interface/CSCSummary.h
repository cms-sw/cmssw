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

#define V_NULL     -1
#define V_TRUE     1
#define V_FALSE    0

class CSCSummary {

  public:

    CSCSummary();

    void Read(TH1*& h1);
    void ReadChambers(TH2*& h2);

    void Write(TH1*& h1);
    void Write(TH1*& h1, const unsigned int station);

    void SetValue(const int value);
    void SetValue(const unsigned int side, const int value);
    void SetValue(const unsigned int side, const unsigned int station, const int value);
    void SetValue(const unsigned int side, const unsigned int station, const unsigned int ring, const int value);
    void SetValue(const unsigned int side, const unsigned int station, const unsigned int ring, const unsigned int chamber, const int value);
    void SetValue(const unsigned int side, const unsigned int station, const unsigned int ring, const unsigned int chamber, const unsigned int cfeb, const int value);
    void SetValue(const unsigned int side, const unsigned int station, const unsigned int ring, const unsigned int chamber, const unsigned int cfeb, const unsigned int hv, const int value);

    const int GetValue(const unsigned int side, const unsigned int station, const unsigned int ring, const unsigned int chamber, const unsigned int cfeb, const unsigned int hv);

  private:

    bool ChamberCoords(const unsigned int x, const unsigned int y, unsigned int& side, unsigned int& station, unsigned int& ring, unsigned int& chamber);

    int map[N_SIDES][N_STATIONS][N_RINGS][N_CHAMBERS][N_CFEBS][N_HVS];

};
