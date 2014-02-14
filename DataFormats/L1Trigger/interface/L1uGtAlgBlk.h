#ifndef L1Trigger_L1uGtAlgBlk_h
#define L1Trigger_L1uGtAlgBlk_h

/**
* \class L1uGtAlgBlk
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


class L1uGtAlgBlk;
typedef BXVector<L1uGtAlgBlk> L1uGtAlgBxCollection;

// class interface

class L1uGtAlgBlk
{

    

public:
    /// constructors
    L1uGtAlgBlk(); // empty constructor, all members set to zero;

    L1uGtAlgBlk(int orbitNr, int bxNr, int bxInEvent);

    /// destructor
    virtual ~L1uGtAlgBlk();


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

    /// Copy vectors words
    void copyInitialToPrescaled() { m_algoDecisionPreScaled   = m_algoDecisionInitial; }
    void copyPrescaledToFinal() { m_algoDecisionFinal   = m_algoDecisionPreScaled; }

    /// Set decision bits
    void setAlgoDecisionInitial(int bit, bool val);
    void setAlgoDecisionPreScaled(int bit, bool val);
    void setAlgoDecisionFinal(int bit, bool val);

    /// Get decision bits
    bool getAlgoDecisionInitial(unsigned int bit) const;
    bool getAlgoDecisionPreScaled(unsigned int bit) const;
    bool getAlgoDecisionFinal(unsigned int bit) const;

    /// reset the content of a L1uGtAlgBlk
    void reset();

    /// pretty print the content of a L1uGtAlgBlk
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

   
    std::vector<bool> m_algoDecisionInitial;
    std::vector<bool> m_algoDecisionPreScaled;
    std::vector<bool> m_algoDecisionFinal;



};

#endif /*L1Trigger_L1uGtAlgBlk_h*/
