#ifndef L1GlobalTrigger_L1TriggerObject_h
#define L1GlobalTrigger_L1TriggerObject_h

/**
 * \class L1TriggerObject
 * 
 * 
 * 
 * Description: abstract base class for L1 trigger objects   
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date:$
 * $Revision:$
 *
 */

// system include files
#include <string>

// user include files
//   base class

// forward declarations


class L1TriggerObject
{

public:

    /// return the name of the object class as string 
    virtual std::string name() const = 0;

    /// empty candidate  - true if object not initialized
    virtual bool empty() const = 0;

};

#endif 
