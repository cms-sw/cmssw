#ifndef L1GlobalTrigger_L1GtLogicalParser_h
#define L1GlobalTrigger_L1GtLogicalParser_h

/**
 * \class L1GtLogicParser
 * 
 * 
 * 
 * Description: parses the logical and numerical expressions for an algorithm 
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
#include <vector>

#include <iosfwd>

// user include files

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
    L1GtLogicParser(const std::string algoLogicalExpressionVal,
                    const std::string algoNumericalExpressionVal);

    /// destructor
    virtual ~L1GtLogicParser();

public:

    /// return the index of the condition in the logical (numerical) expression
    /// this index can then be associated with the index of the CombinationsInCond
    /// in std::vector<CombinationsInCond> of that object map
    int conditionIndex(const std::string condNameVal);

    /// return the name of the (iCondition)th condition in the algorithm logical expression 
    std::string conditionName(const int iCondition);

    /// return the result for a condition in an algorithm
    bool conditionResult(const std::string condNameVal);


private:

    // logical expression for the algorithm
    std::string m_algoLogicalExpression;

    // numerical expression for the algorithm
    // (logical expression with conditions replaced with the actual values)
    std::string m_algoNumericalExpression;

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

    virtual OperationType getOperation(const std::string& tokenString,
                                       OperationType lastOperation, TokenRPN& rpnToken);

    static const struct OperationRule m_operationRules[];

};

#endif /*L1GlobalTrigger_L1GtLogicalParser_h*/
