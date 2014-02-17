//-------------------------------------------------
//
/**  \class L1MuDTSEU
 *
 *   Single Extrapolation Unit:
 *
 *   performs for a given start track segment and a 
 *   given extrapolation type extrapolations 
 *   to all possible target track segments (12 or 6)
 *
 *
 *   $Date: 2008/02/18 17:38:04 $
 *   $Revision: 1.4 $
 *
 *   N. Neumeister            CERN EP
 */
//
//--------------------------------------------------
#ifndef L1MUDT_SEU_H
#define L1MUDT_SEU_H

//---------------
// C++ Headers --
//---------------

#include <utility>
#include <vector>
#include <bitset>

//----------------------
// Base Class Headers --
//----------------------

#include "L1Trigger/DTTrackFinder/interface/L1AbstractProcessor.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "CondFormats/L1TObjects/interface/L1MuDTExtParam.h"
class L1MuDTSectorProcessor;
class L1MuDTTrackSegPhi;
class L1MuDTEUX;
class L1MuDTERS;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class L1MuDTSEU : public L1AbstractProcessor {

  public:

    /// constructor
    L1MuDTSEU(const L1MuDTSectorProcessor& sp, Extrapolation ext, unsigned int tsId );

    /// destructor
    virtual ~L1MuDTSEU();

    /// run SEU
    virtual void run(const edm::EventSetup& c);
   
    /// reset SEU
    virtual void reset();

    /// reset single extrapolation
    void reset(unsigned int relAdr);
    
    /// load data into the SEU 
    inline void load(const L1MuDTTrackSegPhi* startTS) { m_startTS = startTS; }

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
    inline const std::vector<L1MuDTEUX*>& eux() const { return m_EUXs; }
    
    /// return pointer to Extrapolation Result Selector
    inline const L1MuDTERS* ers() const { return m_ERS; }

  private:

    const L1MuDTSectorProcessor& m_sp;
    Extrapolation                m_ext;         // Extrapolation type
    unsigned int                 m_startTS_Id;  // rel. address of start TS
  
    const L1MuDTTrackSegPhi*     m_startTS;     // start track segment
    std::vector<L1MuDTEUX*>      m_EUXs;        // vector of Extrapolators
    L1MuDTERS*                   m_ERS;         // Extrapolation Result Selector

    std::bitset<12>              m_EXtable;     // Extrapolator table
    std::bitset<12>              m_QStable;     // Quality Selector table         
};

#endif
