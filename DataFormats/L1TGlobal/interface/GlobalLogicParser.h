#ifndef L1TGlobal_GlobalLogicParser_h
#define L1TGlobal_GlobalLogicParser_h

// system include files
#include <string>
#include <vector>
#include <list>
#include <map>

#include <iosfwd>

// user include files

// forward declarations

// class declaration
class GlobalLogicParser
{

public:

    struct OperandToken
    {
        std::string tokenName;
        int tokenNumber;
        bool tokenResult;
    };

    enum OperationType {
        OP_NULL=1,
        OP_INVALID=2,
        OP_AND=4,
        OP_OR=8,
        OP_NOT=16,
        OP_OPERAND=32,
        OP_OPENBRACKET=64,
        OP_CLOSEBRACKET=128,
        OP_XOR=256
    };

    struct TokenRPN
    {
        OperationType operation;             // type of operation: AND, OR, NOT or OPERAND
        std::string   operand;               // a possible operand
    };

    typedef std::vector<TokenRPN> RpnVector;

public:

    /// constructor(s)

    ///   default constructor
    GlobalLogicParser();

    ///   from the RPN vector and the operand token vector
    ///   no checks for consistency, empty logical and numerical expressions
    ///   requires special care when used
    GlobalLogicParser(const RpnVector&, const std::vector<OperandToken>&);

    ///   from a constant logical expression
    ///   numerical expression will be empty
    GlobalLogicParser(const std::string& logicalExpressionVal);

    //   from a non-constant logical expression - add/remove spaces if needed
    //   numerical expression will be empty
    GlobalLogicParser(std::string& logicalExpressionVal);

    ///   from a logical and a numerical expression
    GlobalLogicParser(const std::string logicalExpressionVal,
                    const std::string numericalExpressionVal);

    ///   from a logical and a numerical expression
    ///   no checks for correctness - use it only after the correctness was tested
    GlobalLogicParser(const std::string& logicalExpressionVal,
                    const std::string& numericalExpressionVal,
                    const bool dummy);

    /// destructor
    virtual ~GlobalLogicParser();

public:

    /// return the logical expression
    inline std::string logicalExpression() const { return m_logicalExpression; }

    /// check a logical expression for correctness - add/remove spaces if needed
    bool checkLogicalExpression(std::string&);

    /// return the numerical expression
    inline std::string numericalExpression() const { return m_numericalExpression; }

public:

    /// build the rpn vector
    bool buildRpnVector(const std::string&);

    /// clear possible old rpn vector
    void clearRpnVector();

    /// return the RPN vector
    inline RpnVector rpnVector() const { return m_rpnVector; }

    /// build from the RPN vector the operand token vector
    /// dummy tokenNumber and tokenResult
    void buildOperandTokenVector();

    /// return the vector of operand tokens
    inline std::vector<OperandToken>& operandTokenVector() { return m_operandTokenVector; }
    inline const std::vector<OperandToken>& operandTokenVector() const { return m_operandTokenVector; }

public:

    /// return the position index of the operand in the logical expression
    int operandIndex(const std::string& operandNameVal) const;

    /// return the name of the (iOperand)th operand in the logical expression
    std::string operandName(const int iOperand) const;

    /// return the result for an operand with name operandNameVal
    /// in the logical expression using the operand token vector
    bool operandResult(const std::string& operandNameVal) const;

    /// return the result for an operand with tokenNumberVal
    /// using the operand token vector
    bool operandResult(const int tokenNumberVal) const;

    /// return the result for the logical expression
    /// require a proper operand token vector
    virtual const bool expressionResult() const;

    /// return the result for an operand with name operandNameVal
    /// in the logical expression using a numerical expression
    bool operandResultNumExp(const std::string& operandNameVal) const;

    /// return the result for an operand with index iOperand
    /// in the logical expression using a numerical expression
    bool operandResultNumExp(const int iOperand) const;

    /// build from the RPN vector the operand token vector
    /// using a numerical expression
    void buildOperandTokenVectorNumExp();

    /// return the result for the logical expression
    /// require a proper numerical expression
    virtual const bool expressionResultNumExp() const;

    /// convert the logical expression composed with names to
    /// a logical expression composed with int numbers using
    /// a (string, int)  map
    void convertNameToIntLogicalExpression(
        const std::map<std::string, int>& nameToIntMap);

    /// convert a logical expression composed with integer numbers to
    /// a logical expression composed with names using a map (int, string)

    void convertIntToNameLogicalExpression(const std::map<int, std::string>& intToNameMap);

    /// return the list of operand tokens for the logical expression
    /// which are to be used as seeds
    std::vector<GlobalLogicParser::OperandToken> expressionSeedsOperandList();


protected:


    struct OperationRule
    {
        const char* opString;
        int         opType;
        int         forbiddenLastOperation;    // int for bitmask of forbidden operations
    };


    virtual OperationType getOperation(const std::string& tokenString,
                                       OperationType lastOperation, TokenRPN& rpnToken) const;

    /// get the rule entry to an operation type
    const OperationRule* getRuleFromType(OperationType t);

    static const struct OperationRule m_operationRules[];

protected:

    /// add spaces before and after parantheses
    void addBracketSpaces(const std::string&, std::string&);

    /// set the logical expression - check for correctness the input string
    bool setLogicalExpression(const std::string&);

    /// set the numerical expression (the logical expression with each operand
    /// replaced with the value) from a string
    /// check also for correctness the input string
    bool setNumericalExpression(const std::string&);

protected:

    /// logical expression to be parsed
    std::string m_logicalExpression;

    /// numerical expression
    /// (logical expression with operands replaced with the actual values)
    std::string m_numericalExpression;

    /// RPN vector - equivalent to the logical expression
    RpnVector m_rpnVector;

    /// vector of operand tokens
    std::vector<OperandToken> m_operandTokenVector;


};

#endif /*L1TGlobal_GtLogicParser_h*/
