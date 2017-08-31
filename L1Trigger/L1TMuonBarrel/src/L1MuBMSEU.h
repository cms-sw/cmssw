//-------------------------------------------------
//
/**  \class L1MuBMSEU
 *
 *   Single Extrapolation Unit:
 *
 *   performs for a given start track segment and a
 *   given extrapolation type extrapolations
 *   to all possible target track segments (12 or 6)
 *
 *
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUBM_SEU_H
#define L1MUBM_SEU_H

//---------------
// C++ Headers --
//---------------

#include <utility>
#include <vector>
#include <bitset>

//----------------------
// Base Class Headers --
//----------------------

#include "L1Trigger/L1TMuonBarrel/interface/L1AbstractProcessor.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "CondFormats/L1TObjects/interface/L1MuDTExtParam.h"
class L1MuBMSectorProcessor;
class L1MuBMTrackSegPhi;
class L1MuBMEUX;
class L1MuBMERS;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuBMSEU : public L1AbstractProcessor {

  public:

    /// constructor
    L1MuBMSEU(const L1MuBMSectorProcessor& sp, Extrapolation ext, unsigned int tsId );

    /// destructor
    ~L1MuBMSEU() override;

    /// run SEU
    void run(const edm::EventSetup& c) override;

    /// reset SEU
    void reset() override;

    /// reset single extrapolation
    void reset(unsigned int relAdr);

    /// load data into the SEU
    inline void load(const L1MuBMTrackSegPhi* startTS) { m_startTS = startTS; }

    /// return Extrapolator table
    const std::bitset<12>& exTable() const { return m_EXtable; }

    /// return Quality Sorter table
    const std::bitset<12>& qsTable() const { return m_QStable; }

    /// return number of successful extrapolations
    int numberOfExt() const;

    /// return extrapolation type
    inline Extrapolation ext() const { return m_ext; }

    /// return start track segment identifier (relative address)
    inline unsigned int tsId() const { return m_startTS_Id; }

    /// is it a own wheel Single Extrapolation Unit
    inline bool isOwnWheelSEU() const { return ( m_startTS_Id == 0 || m_startTS_Id == 1 ); }

    /// is it a next wheel Single Extrapolation Unit
    inline bool isNextWheelSEU() const { return ( m_startTS_Id == 2 || m_startTS_Id == 3 ); }

    /// return pointer to an Extrapolator
    inline const std::vector<L1MuBMEUX*>& eux() const { return m_EUXs; }

    /// return pointer to Extrapolation Result Selector
    inline const L1MuBMERS* ers() const { return m_ERS; }

  private:

    const L1MuBMSectorProcessor& m_sp;
    Extrapolation                m_ext;         // Extrapolation type
    unsigned int                 m_startTS_Id;  // rel. address of start TS

    const L1MuBMTrackSegPhi*     m_startTS;     // start track segment
    std::vector<L1MuBMEUX*>      m_EUXs;        // vector of Extrapolators
    L1MuBMERS*                   m_ERS;         // Extrapolation Result Selector

    std::bitset<12>              m_EXtable;     // Extrapolator table
    std::bitset<12>              m_QStable;     // Quality Selector table
};

#endif
