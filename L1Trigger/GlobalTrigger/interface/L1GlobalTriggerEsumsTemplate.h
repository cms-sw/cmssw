#ifndef GlobalTrigger_L1GlobalTriggerEsumsTemplate_h 
#define GlobalTrigger_L1GlobalTriggerEsumsTemplate_h
/**
 * \class L1GlobalTriggerEsumsTemplate
 * 
 * 
 * 
 * Description: 
 * Implementation: Single particle chip: energy sums conditions
 *    <TODO: enter implementation details>
 *   
 * \author: M.Eder               - HEPHY Vienna - ORCA version 
 * \author: Vasile Mihai Ghete   - HEPHY Vienna - CMSSW version 
 * 
 * $Date:$
 * $Revision:$
 *
 */

// system include files
#include <string>

// user include files

//   base class
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerConditions.h"

// forward declarations

//class interface
class L1GlobalTriggerEsumsTemplate : public L1GlobalTriggerConditions
{

public:

    // constructor
    L1GlobalTriggerEsumsTemplate(const std::string &name);
    
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
        ETM = 0,
        ETT,
        HTT
    };  

    // set functions
    void setConditionParameter(const ConditionParameter *conditionp, SumType st);
    
    // get functions
    inline const ConditionParameter* getConditionParameter() const { return &(this->p_conditionparameter); }
        
    virtual const bool blockCondition() const;    
    
    // print thresholds
    void printThresholds() const;
              
 private:
    
    // what type of energy sum is this condition for
    SumType p_sumtype;

    // variables containing the parameters
    ConditionParameter p_conditionparameter;
    
    // copy function for copy constructor and operator=
    void copy( const L1GlobalTriggerEsumsTemplate& cp);

};

#endif 
