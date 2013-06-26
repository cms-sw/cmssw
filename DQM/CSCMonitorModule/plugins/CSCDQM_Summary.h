/*
 * =====================================================================================
 *
 *       Filename:  Summary.h
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

#ifndef CSCDQM_Summary_H
#define CSCDQM_Summary_H

#include <TH2.h>
#include <math.h>
#include <vector>
#include <bitset>
#include <iostream>

#include "CSCDQM_Detector.h"
#include "CSCDQM_Logger.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#define HWSTATUSBITSETSIZE    14
#define HWSTATUSERRORBITS     0xffc
#define HWSTATUSEQUALS(s, m)  (((std::bitset<HWSTATUSBITSETSIZE>) m & s) == m)
#define HWSTATUSANY(s, m)     (((std::bitset<HWSTATUSBITSETSIZE>) m & s).any())
#define HWSTATUSANYERROR(s)   (HWSTATUSANY(s, HWSTATUSERRORBITS))

#define NTICS                 100

namespace cscdqm {

/**
 * @brief Hardware Status Bit values used in Summary efficiency calculation
 */
enum HWStatusBit {

  DATA,         /// Data available (reporting)
  MASKED,       /// HW element was masked out (not in readout)

  HOT,          /// HW element is hot by comparing with reference histogram 
  COLD,         /// HW element is cold comparing with reference histogram

  FORMAT_ERR,   /// Format errors
  L1SYNC_ERR,   /// L1A out of sync errors
  FIFOFULL_ERR, /// DMB FIFO full error
  INPUTTO_ERR,  /// DMB Input timeout error

  NODATA_ALCT,  /// No ALCT data
  NODATA_CLCT,  /// No CLCT data
  NODATA_CFEB,  /// No CFEB data
  CFEB_BWORDS,  /// Data with CFEB BWORDS

  STANDBY,      /// HW element in standby mode
  WAS_ON        /// HW element ever in the run was ON (off standby mode)

};

/**
 * @brief  Hardware Status Bits structure used in Summary efficiency
 * calculation and storage
 */
typedef std::bitset<HWSTATUSBITSETSIZE> HWStatusBitSet;

/**
 * @class Summary
 * @brief Hardware and Physics Efficiency data structures and routines 
 */
class Summary {

  public:

    Summary();
    ~Summary();

    void Reset();

    const Detector getDetector() const { return detector; }

    void ReadReportingChambers(const TH2*& h2, const double threshold = 1.0);
    void ReadReportingChambersRef(const TH2*& h2, const TH2*& refh2, const double cold_coef = 0.1, const double cold_Sfail = 5.0, const double hot_coef = 2.0, const double hot_Sfail = 5.0);
    void ReadErrorChambers(const TH2*& evs, const TH2*& err, const HWStatusBit bit, const double eps_max = 0.1, const double Sfail = 5.0);

    const unsigned int setMaskedHWElements(std::vector<std::string>& tokens);

    void Write(TH2*& h2, const unsigned int station) const;
    void WriteMap(TH2*& h2);
    void WriteChamberState(TH2*& h2, const int mask, const int value = 1, const bool reset = true, const bool op_any = false) const;

    void ReSetValue(const HWStatusBit bit);
    void ReSetValue(Address adr, const HWStatusBit bit);
    void SetValue(const HWStatusBit bit, const int value = 1);
    void SetValue(Address adr, const HWStatusBit bit, const int value = 1);

    const HWStatusBitSet GetValue(Address adr) const;
    const int IsPhysicsReady(const unsigned int px, const unsigned int py);
    //const int IsPhysicsReady(const float xmin, const float xmax, const float ymin, const float ymax) const;

    const double GetEfficiencyHW() const;
    const double GetEfficiencyHW(const unsigned int station) const;
    const double GetEfficiencyHW(Address adr) const; 
    const double GetEfficiencyArea(const unsigned int station) const; 
    const double GetEfficiencyArea(Address adr) const; 

    bool isChamberStandby(unsigned int side, unsigned int station, unsigned int ring, unsigned int chamber) const;
    bool isChamberStandby(CSCDetId cid) const;

  private:

    const bool ChamberCoordsToAddress(const unsigned int x, const unsigned int y, Address& adr) const;
    const bool ChamberAddressToCoords(const Address& adr, unsigned int& x, unsigned int& y) const;
    const double GetReportingArea(Address adr) const; 

    /** Atomic HW element status matrix */
    HWStatusBitSet map[N_SIDES][N_STATIONS][N_RINGS][N_CHAMBERS][N_LAYERS][N_CFEBS][N_HVS];

    /** Detector routines object */
    Detector detector;

};

}

#endif
