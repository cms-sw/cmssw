#ifndef L1GlobalTrigger_L1GtLogicParser_h
#define L1GlobalTrigger_L1GtLogicParser_h

/**
 * \class L1GtLogicParser
 * 
 * 
 * Description: parses a logical expression, with predefined operators.  
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
#include <vector>
#include <list>
#include <map>

#include <iosfwd>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

// forward declarations
class L1GlobalTriggerObjectMap;

// class declaration
class L1GtLogicParser
{

public:

    /// constructor(s)

    ///   from an object map
    L1GtLogicParser(const L1GlobalTriggerObjectMap& );

    ///   from a logical and a numerical expression
    L1GtLogicParser(const std::string logicalExpressionVal,
                    const std::string numericalExpressionVal);

    ///   from a logical expression,a DecisionWord and a map (string, int)
    ///   should be used for logical expressions with algorithms
    ///   the map convert from algorithm name to algorithm bit number, if needed
    L1GtLogicParser(const std::string&,
                    const DecisionWord&,
                    const std::map<std::string,int>&);

    /// destructor
    virtual ~L1GtLogicParser();

public:

    /// return the position index of the operand in the logical expression
    int operandIndex(const std::string operandNameVal) const;

    /// return the name of the (iOperand)th operand in the logical expression
    std::string operandName(const int iOperand) const;

    /// return the result for an operand with name operandNameVal
    /// from the logical expression using a numerical expression
    bool operandResult(const std::string operandNameVal) const;

    /// return the result for an operand with index iOperand
    /// in the logical expression using a numerical expression
    bool operandResult(const int iOperand) const;

    /// return the result for the logical expression
    virtual const bool expressionResult() const;

    /// return the list of operands for the logical expression
    virtual const std::list<int> expressionOperandList() const;

    /// return the list of indices of the operands in the logical expression
    virtual const std::list<int> expressionOperandIndexList() const;

private:

    enum OperationType {
        OP_NULL=1,
        OP_INVALID=2,
        OP_AND=4,
        OP_OR=8,
        OP_NOT=16,
        OP_OPERAND=32,
        OP_OPENBRACKET=64,
        OP_CLOSEBRACKET=128
    };

    struct OperationRule
    {
        const char* opString;
        int         opType;
        int         forbiddenLastOperation;    // int for bitmask of forbidden operations
    };

    typedef struct
    {
        OperationType operation;             // type of operation: AND, OR, NOT or OPERAND
        std::string   operand;               // a possible operand
    }
    TokenRPN;

    typedef std::vector<TokenRPN> RpnVector;

    virtual OperationType getOperation(const std::string& tokenString,
                                       OperationType lastOperation, TokenRPN& rpnToken) const;

    /// get the rule entry to an operation type
    const OperationRule* getRuleFromType(OperationType t);

    /// build the rpn vector
    int buildRpnVector(const std::string&);

    /// clear possible old rpn vector
    void clearRpnVector();

    static const struct OperationRule m_operationRules[];

private:

    /// add spaces before and after parantheses
    void addBracketSpaces(const std::string&, std::string&);

    /// set the logical expression - check for correctness the input string
    int setLogicalExpression(const std::string&);

    /// set the numerical expression (the logical expression with each operand
    /// replaced with the value) from a string
    /// check also for correctness the input string
    int setNumericalExpression(const std::string&);

    /// convert the logical expression composed with algorithm bits into a
    /// numerical expression using values from DecisionWord.
    /// the map convert from algorithm name to algorithm bit number, if needed
    int setNumericalExpression(const DecisionWord& decisionWordVal,
                               const std::map<std::string, int>& algoMap);

    /// return the result for an operand from a logical expression
    /// from a decision word
    bool operandResultDecisionWord(const std::string& operandIdentifier,
                                   const DecisionWord& decisionWordVal,
                                   const std::map<std::string, int>& algoMap);

private:

    /// logical expression to be parsed
    std::string m_logicalExpression;

    /// numerical expression
    /// (logical expression with operands replaced with the actual values)
    std::string m_numericalExpression;

    /// RPN vector
    RpnVector m_rpnVector;


};

#endif /*L1GlobalTrigger_L1GtLogicParser_h*/
