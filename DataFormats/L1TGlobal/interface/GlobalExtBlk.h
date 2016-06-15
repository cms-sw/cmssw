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


// forward declarations

class GlobalExtBlk;
typedef BXVector<GlobalExtBlk> GlobalExtBlkBxCollection;
  
// class interface

class GlobalExtBlk
{

public:
    /// constructors
    GlobalExtBlk(); // empty constructor, all members set to zero;

    /// destructor
    virtual ~GlobalExtBlk();


public:
    const static unsigned int maxExternalConditions = 256;
    
    /// Set decision bits
    void setExternalDecision(unsigned int bit, bool val);

    /// Get decision bits
    bool getExternalDecision(unsigned int bit) const;

    /// reset the content of a GlobalExtBlk
    void reset();

    /// pretty print the content of a GlobalExtBlk
    void print(std::ostream& myCout) const;


private:

   
    std::vector<bool> m_extDecision;


};

#endif /*L1Trigger_GlobalExtBlk_h*/
