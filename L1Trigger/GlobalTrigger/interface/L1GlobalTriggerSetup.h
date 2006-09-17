#ifndef GlobalTrigger_L1GlobalTriggerSetup_h
#define GlobalTrigger_L1GlobalTriggerSetup_h

/**
 * \class L1GlobalTriggerSetup
 * 
 * 
 * 
 * Description: L1 Global Trigger Setup 
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <string>

// user include files
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// forward declarations
class L1GlobalTrigger;
class L1GlobalTriggerConfig;

// class interface
class L1GlobalTriggerSetup
{
public:

    // constructor
	L1GlobalTriggerSetup(L1GlobalTrigger&, const edm::ParameterSet&);
    
    // destructor
	virtual ~L1GlobalTriggerSetup();
    
public:

    // maximum number of blocks per trigger object  
    static const int MaxItem = 64;

 
    static const edm::ParameterSet* getParameterSet() { return m_pSet; }
    
    virtual void setTriggerMenu(std::string&);
    
    const L1GlobalTriggerConfig* gtConfig() const { return m_gtConfig; }
    
private:
    
    L1GlobalTrigger& m_GT;

    static const edm::ParameterSet* m_pSet;
    static L1GlobalTriggerConfig* m_gtConfig;
    
};

#endif /*GlobalTrigger_L1GlobalTriggerSetup_h*/
