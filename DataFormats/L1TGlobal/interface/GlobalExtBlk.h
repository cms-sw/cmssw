#ifndef L1Trigger_GlobalExtBlk_h
#define L1Trigger_GlobalExtBlk_h

/**
* \class GlobalExtBlk
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
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"

// forward declarations

class GlobalExtBlk;
typedef BXVector<GlobalExtBlk> GlobalExtBlkBxCollection;

// class interface

class GlobalExtBlk
{

public:
    /// constructors
    GlobalExtBlk(); // empty constructor, all members set to zero;

    GlobalExtBlk(int orbitNr, int bxNr, int bxInEvent);

    /// destructor
    virtual ~GlobalExtBlk();


public:

    /// set simple members
    void setOrbitNr(int orbNr)     { m_orbitNr   = orbNr; }
    void setbxNr(int bxNr)         { m_bxNr      = bxNr; }
    void setbxInEventNr(int bxNr)  { m_bxInEvent = bxNr; }
    void setFinalOR(int fOR)       { m_finalOR   = fOR; }

    /// get simple members
    inline const int getOrbitNr() const     { return m_orbitNr; }
    inline const int getbxNr() const        { return m_bxNr; }
    inline const int getbxInEventNr() const { return m_bxInEvent; }
    inline const int getFinalOR() const     { return m_finalOR; }

    /// Set decision bits
    void setExternalDecision(int bit, bool val);

    /// Get decision bits
    bool getExternalDecision(unsigned int bit) const;

    /// reset the content of a GlobalExtBlk
    void reset();

    /// pretty print the content of a GlobalExtBlk
    void print(std::ostream& myCout) const;


private:

    /// orbit number
    int m_orbitNr;

    /// bunch cross number of the actual bx
    int m_bxNr;

    /// bunch cross in the GT event record (E,F,0,1,2)
    int m_bxInEvent;

    // finalOR 
    int m_finalOR;

   
    std::vector<bool> m_extDecision;


};

#endif /*L1Trigger_GlobalExtBlk_h*/
