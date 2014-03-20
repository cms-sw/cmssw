#ifndef L1Trigger_L1uGtRecBlk_h
#define L1Trigger_L1uGtRecBlk_h

/**
* \class L1uGtRecBlk
*
*
* Description: L1 micro Global Trigger - Block holding Algorithm Information
*
* Implementation:
* <TODO: enter implementation details>
*
* \author: Brian Winer - Ohio State
*
*
*/

// system include files
#include <vector>
#include <iostream>
#include <iomanip>

// user include files
#include "FWCore/Utilities/interface/typedefs.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"

// forward declarations
class L1uGtRecBlk;
typedef std::vector<L1uGtRecBlk> L1uGtRecBxCollection;

// class interface

class L1uGtRecBlk
{

public:
    /// constructors
    L1uGtRecBlk(); // empty constructor, all members set to zero;

    L1uGtRecBlk(int ver, int bxAlg, int bxExt, int bxMu, int bxCal, int psIndex,
                cms_uint64_t orb, int bxNr, int lumiS, int uGtNr);

    /// destructor
    virtual ~L1uGtRecBlk();


public:

    /// set simple members
    void setFirmwareVer(int ver)     { m_firmVersion          = ver; }
    void setTotBxAlg(int nr)         { m_totBxInEvent_alg     = nr; }
    void setTotBxExt(int nr)         { m_totBxInEvent_ext     = nr; }
    void setTotBxMuData(int nr)      { m_totBxInEvent_muData  = nr; }
    void setTotBxCalData(int nr)     { m_totBxInEvent_calData = nr; }
    void setPreScIndex(int nr)       { m_prescaleIndex        = nr; }
    
    void setTrigNr( cms_uint64_t nr)  { m_triggerNr   = nr; }
    void setOrbitNr( cms_uint64_t nr) { m_orbitNr     = nr; }
    
    void setbxNr(int nr)           { m_bxNr        = nr; }
    void setLumiSec(int ls)        { m_lumiSection = ls; }
    void setuGtNr( int nr)         { m_internalEvt = nr; }


    void reset();
    void print(std::ostream& myCout) const;
    
private:

    /// Firmware Version
    int m_firmVersion;
    
    /// Number of Bx Stored with Record
    int m_totBxInEvent_alg;
    int m_totBxInEvent_ext;
    int m_totBxInEvent_muData;
    int m_totBxInEvent_calData;

    /// Prescale Index
    int m_prescaleIndex;
    
    /// Trigger Number
    cms_uint64_t m_triggerNr;
    
    /// orbit number
    cms_uint64_t m_orbitNr;

    /// bunch cross number of the actual bx
    int m_bxNr;

    /// Luminosity Segment
    int m_lumiSection;
    
    /// internal GT Event Count
    int m_internalEvt;

    // finalOR (Also Stored with Algorithm Bits)
    std::vector<int> m_finalOR;

};

#endif /*L1Trigger_L1uGtRecBlk_h*/
