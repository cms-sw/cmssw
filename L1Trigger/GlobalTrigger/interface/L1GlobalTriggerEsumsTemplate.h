#ifndef GlobalTrigger_L1GlobalTriggerEsumsTemplate_h 
#define GlobalTrigger_L1GlobalTriggerEsumsTemplate_h

/**
 * \class L1GlobalTriggerEsumsTemplate
 * 
 * 
 * Description: Single particle chip: energy sums conditions.  
 *
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
#include <iosfwd>

// user include files

//   base class
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerConditions.h"

// forward declarations
class L1GlobalTrigger;

//class interface
class L1GlobalTriggerEsumsTemplate : public L1GlobalTriggerConditions
{

public:

    // constructor
    L1GlobalTriggerEsumsTemplate(const L1GlobalTrigger&, const std::string&);
    
    // copy constructor
    L1GlobalTriggerEsumsTemplate( const L1GlobalTriggerEsumsTemplate& );

    // destructor
    virtual ~L1GlobalTriggerEsumsTemplate();
    
    // assign operator
    L1GlobalTriggerEsumsTemplate& operator= (const L1GlobalTriggerEsumsTemplate&);

public:

    ///parameters
    typedef struct
    {
        unsigned int et_threshold;
        bool en_overflow;
        unsigned int phi;           // only for etm
    } ConditionParameter;
      
    enum SumType {
        ETM_ST = ETM,
        ETT_ST = ETT,
        HTT_ST = HTT
    };  

    // set functions
    void setConditionParameter(const ConditionParameter *conditionp, SumType st);
    
    // get functions
    inline const ConditionParameter* getConditionParameter() const { return &(this->p_conditionparameter); }
        
    virtual const bool blockCondition() const;    
    
    // print thresholds
    void printThresholds(std::ostream& myCout) const;
              
 private:
    
    // what type of energy sum is this condition for
    SumType p_sumtype;

    // variables containing the parameters
    ConditionParameter p_conditionparameter;
    
    // copy function for copy constructor and operator=
    void copy( const L1GlobalTriggerEsumsTemplate& cp);

};

#endif 
