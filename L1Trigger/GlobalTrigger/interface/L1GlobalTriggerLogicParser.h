#ifndef GlobalTrigger_L1GlobalTriggerLogicParser_h
#define GlobalTrigger_L1GlobalTriggerLogicParser_h

/**
 * \class L1GlobalTriggerLogicParser
 * 
 * 
 * 
 * Description: class for parsing logic expressions 
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

// user include files
//   base class
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerConditions.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerConfig.h"

// forward declarations
class L1GlobalTrigger;

// class declaration 
class L1GlobalTriggerLogicParser : public L1GlobalTriggerConditions
{

public:

    /// constructor
    L1GlobalTriggerLogicParser(const L1GlobalTrigger&, const std::string& name);

    /// destructor
    virtual ~L1GlobalTriggerLogicParser();

    ///
    virtual const bool blockCondition() const;

    /// get the expression with the results inserted for the operands
    virtual std::string getNumericExpression();
    
    /// set the expression and build the parse tree
    int setExpression( const std::string& expression, 
        L1GlobalTriggerConfig::ConditionsMap* operandmap, unsigned int nummap=1);
    
    /// print thresholds
    virtual void printThresholds() const;

private:

    enum OperationType {    // the values need to be bitwise for the operationrules
        OP_NULL=1,          // null element
        OP_INVALID=2,
        OP_AND=4,
        OP_OR=8,
        OP_NOT=16,
        OP_OPERAND=32,			
        OP_OPENBRACKET=64,		
        OP_CLOSEBRACKET=128		
    };

    struct OperationRule {
        const char* opstr;
        int  optype;
        int forbidden_lastoperation;    // int for bitmask of forbidden operations
    };

    static const struct OperationRule p_operationrules[];
    
    //defines a token in reverse polish notation
    /*class TokenRPN  {
      public:
      OperationType operation;		//The type of operation in this node. (AND, OR, NOT or OPERAND)
      L1GlobalTriggerConditions* operand;	//a possible operand 

      ///we need to define constructors and copy-operation to use this with vector and stack
      //constructor
      TokenRPN() { operation=OP_NULL; operand=0; }
      ///copy-constructor
      TokenRPN( const TokenRPN& cp ) { operation=cp.operation; operand=cp.operand; }
      ///copy assignmend
      const TokenRPN& operator= ( const TokenRPN& cp) { operation=cp.operation; operand=cp.operand; return (*this); }
    };*/

    typedef struct {
        OperationType operation;    // the type of operation in this node. (AND, OR, NOT or OPERAND)
        L1GlobalTriggerConditions* operand;  // a possible operand 
    } TokenRPN;

    typedef vector<TokenRPN> RPNVector;
    RPNVector p_rpnvector;


    /// the expression to be parsed as string
    std::string p_expression;

    /// the postfix expression
    std::string p_rpnexpression;

    /// the reference to the conditionsmap of the operands
    L1GlobalTriggerConfig::ConditionsMap *p_operandmap;
    /// number of operand maps at the reference
    unsigned int p_nummap;

    /// clear the vector that contains the reverse polish notation
    void clearRPNVector();

    /// add spaces before and after brackets
    void addBracketSpaces(const std::string &srcexpression, std::string& dstexpression);

    /// get the operation type and check if it is valid in this scope
    OperationType getOperation(const std::string &tokenstr,
        OperationType lastoperation, TokenRPN& tokenrpn);

    /// try to build the rpn vector 
    int buildRPNVector(const std::string& expression);

    /// build the postfix std::string
    void buildRPNExpression();    

    /// get the rule entry to an operation type
    const OperationRule* getRuleFromType(OperationType t);      
    
};

#endif
