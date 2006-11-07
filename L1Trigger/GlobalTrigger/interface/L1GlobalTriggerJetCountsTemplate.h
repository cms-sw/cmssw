#ifndef GlobalTrigger_L1GlobalTriggerJetCountsTemplate_h 
#define GlobalTrigger_L1GlobalTriggerJetCountsTemplate_h

/**
 * \class L1GlobalTriggerJetCountsTemplate
 * 
 * 
 * 
 * Description: Single particle chip - jet counts conditions
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: M.Eder               - HEPHY Vienna - ORCA version 
 * \author: Vasile Mihai Ghete   - HEPHY Vienna - CMSSW version 
 * 
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <string>
#include <iosfwd>

// user include files
//   base class
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerConditions.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

// forward declarations
class L1GlobalTrigger;

// class declaration
class L1GlobalTriggerJetCountsTemplate : public L1GlobalTriggerConditions 
{

public:

    // constructor
    L1GlobalTriggerJetCountsTemplate(const L1GlobalTrigger&, const std::string&);
  
    // copy constructor
    L1GlobalTriggerJetCountsTemplate( const L1GlobalTriggerJetCountsTemplate& );

    // destructor
    virtual ~L1GlobalTriggerJetCountsTemplate();
  
    // assign operator
    L1GlobalTriggerJetCountsTemplate& operator = (const L1GlobalTriggerJetCountsTemplate&);


public:

    // parameters
    typedef struct
    {
        unsigned int et_threshold;
        unsigned int type;
    } ConditionParameter;
        
    // set functions
    void setConditionParameter(const ConditionParameter *conditionp);
    
    // get functions
    inline const ConditionParameter* getConditionParameter() const { return &(this->p_conditionparameter); }

    virtual const bool blockCondition() const;
        
    // print thresholds
    void printThresholds(std::ostream& myCout) const;

private:

    // variables containing the parameters
    ConditionParameter p_conditionparameter;
   
    // copy function for copy constructor and operator=
    void copy( const L1GlobalTriggerJetCountsTemplate& cp);

    
};

#endif 
