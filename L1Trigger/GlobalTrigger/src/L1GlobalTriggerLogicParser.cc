/**
 * \class L1GlobalTriggerLogicParser
 * 
 * 
 * 
 * Description: see header file 
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: M.Eder               - HEPHY Vienna - ORCA version 
 * \author: Vasile Mihai Ghete   - HEPHY Vienna - CMSSW version 
 * 
 * $Date:$
 * $Revision:$
 *
 */

// this class header
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerLogicParser.h"

// system include files
#include <string>
#include <stack>

// user include files
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerConfig.h"
//#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerPSB.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


// constructor
L1GlobalTriggerLogicParser::L1GlobalTriggerLogicParser(const std::string &name)
    : L1GlobalTriggerConditions(name) {
        
    LogDebug ("Trace") << "****Entering " << __PRETTY_FUNCTION__ 
        << " name= " << p_name << std::endl;
    p_expression.clear();
    p_rpnexpression.clear();
    p_operandmap = NULL;
    p_rpnvector.clear();
    p_nummap = 0;

}

// destructor
L1GlobalTriggerLogicParser::~L1GlobalTriggerLogicParser() {

}

/**
 * clearRPNVector Function to delete the whole RPNVector.
 *
 *
 */

void L1GlobalTriggerLogicParser::clearRPNVector() {

    p_rpnvector.clear();
    
}

/**
 * addBracketSpaces Adds spaces before and after brackets, to make separation easier
 *
 * @param srcexpression The source string with no spaces added.
 * @param dstexpression The destination where the string with whitespaces is written to.
 *
 */

void L1GlobalTriggerLogicParser::addBracketSpaces(const std::string& srcexpression,
    std::string& dstexpression) {
                
    static const std::string brackets="()";	// the brackets to be found

    dstexpression = srcexpression;	// copy the string

    size_t position=0;

    while ( (position = dstexpression.find_first_of(brackets, position)) != std::string::npos ) {
        // add space after if none is there
        if (dstexpression[position + 1] != ' ') {
            dstexpression.insert(position+1, " ");
        }
        // add space befor if none is there
        if (dstexpression[position - 1] != ' ') {
            dstexpression.insert(position, " ");
            position++;
        }
        position++;
    }
} 

/**
 * getOperation Get the operation from a string and check if it is allowed
 *
 * @param tokenstr The string to examine.
 * @param lastoperation The last operation. (Needed to check if the actual operation is valid)
 * @param tokenrpn The destination where the token for postfix notation is written to.
 *
 * @return The Operation type or OP_INVALID, if the operation is not allowed
 *
 */

L1GlobalTriggerLogicParser::OperationType 
    L1GlobalTriggerLogicParser::getOperation(const std::string& tokenstr, 
    OperationType lastoperation, TokenRPN& tokenrpn) {

    OperationType actoperation;

    int i = 0;
    actoperation = OP_OPERAND;	// default value
    while (p_operationrules[i].optype != OP_OPERAND) {
        if (tokenstr == p_operationrules[i].opstr) { 
            actoperation = (OperationType) p_operationrules[i].optype;
            break;
        }
        i++;
    }

    // check if the operation is allowed
    if (p_operationrules[i].forbidden_lastoperation &lastoperation) {
        return OP_INVALID;
    }

    // check if a operand exists in the operandmap
    if (actoperation == OP_OPERAND) {
        
        tokenrpn.operand = NULL;
        for (unsigned int ui = 0; ui < p_nummap; ui++) {
            if (p_operandmap[ui].count(tokenstr) != 0) {	// TODO: unique test?
                tokenrpn.operand = p_operandmap[ui][tokenstr];    // fill in the operand
            }
        }
        
        // if no operand found with this name
        if (tokenrpn.operand==NULL) {
            return OP_INVALID;
        }

    } else {
        tokenrpn.operand=NULL;
    }

    tokenrpn.operation = actoperation;

    // else we got a valid operation
    return actoperation;
}

/**
 * getRuleFromType Looks for the entry in the operation rules 
 *     and returns a reference if it was found
 *
 * @param t The type of the operation.
 *
 * @return The reference to the entry or NULL if the Rule was not found.
 *
 */

const L1GlobalTriggerLogicParser::OperationRule* 
    L1GlobalTriggerLogicParser::getRuleFromType(OperationType t) {


    int i = 0;
    while (p_operationrules[i].optype != t && p_operationrules[i].optype != OP_NULL) {
        i++;
    }

    if (p_operationrules[i].optype == OP_NULL) {
        return NULL;
    }
    
    return &(p_operationrules[i]);
}

/**
 * buildRPNExpression Build the postfix expression string 
 *
 */

void L1GlobalTriggerLogicParser::buildRPNExpression() {

    p_rpnexpression.clear();    // clear a possible old string

    RPNVector::iterator it;     
    for(it = p_rpnvector.begin(); it != p_rpnvector.end(); it++) {
        if (it->operation == OP_OPERAND) {
            p_rpnexpression.append(it->operand->getName());		// an operand is written as its name
        } else {
            p_rpnexpression.append(getRuleFromType(it->operation)->opstr);
        }
        p_rpnexpression.append(" ");	// spaces in between
    }
}

/**
 * getNumericExpression get a string with each operand replaced with 
 *     the value from the operand.
 *
 *
 * @return A string with all operands replaced with their values.
 *
 */


std::string L1GlobalTriggerLogicParser::getNumericExpression() {
    
    // add spaces before and after brackets to the original expression
    std::string exprwithbs;
    addBracketSpaces(p_expression, exprwithbs);

    std::string result;         // result string
    result.clear();

    OperationType actoperation = OP_NULL;
    OperationType lastoperation = OP_NULL;

    std::string tokenstr;
    TokenRPN tokenrpn;    // token for the use of getOperation
        
    istringstream expr_iss(exprwithbs);	// stringstream to seperate all tokens

    while (!expr_iss.eof()) {
        
        expr_iss >> tokenstr;
        if (expr_iss.eof()) {
            break;
        }

        actoperation = getOperation(tokenstr, lastoperation, tokenrpn);
        if (actoperation == OP_INVALID) {
            return result;	// it should never be invalid, because it is checked
        }
        
        if (actoperation != OP_OPERAND) {
            result.append(getRuleFromType(actoperation)->opstr); 
        } else {
            if (tokenrpn.operand->getLastResult()) { // replace the operand with its result
                result.append("1"); // true
            } else {
                result.append("0"); // false
            }
        }
        result.append(" ");         // one whitespace after each token
        lastoperation = actoperation;
    }

    return result;

}

/**
 * buildRPNVector Try to build the postfix notation. 
 *
 * @param expression The expression to be parsed.
 *
 * @return 0 if everything was parsed. -1 if an error occured.
 *
 */

int L1GlobalTriggerLogicParser::buildRPNVector(const std::string& expression) {
    
    istringstream expr_iss(expression);	// stringstream to seperate all tokens

    std::string tokenstr;           // one token
    TokenRPN tokenrpn;              // the token as TokenRPN type
    std::stack<TokenRPN> operatorstack;	// the operatorstack to form the postfix notation
    static const std::string whitespaces=" \r\v\n\t";

    int errorpos;                             // position of a possible error
    OperationType lastoperation = OP_NULL;    // the type of the last operation
    OperationType actoperation  = OP_NULL;    // the actual operation
    
    clearRPNVector();			//clear possible old rpn vector

    while (!expr_iss.eof()) {
        
        expr_iss >> skipws >> ws >> tokenstr;
        // skip the end 
        if (tokenstr.find_first_not_of(whitespaces) == std::string::npos || 
            tokenstr.length() == 0 || expr_iss.eof()) { 
            break; 
        }
        
        actoperation = getOperation(tokenstr, lastoperation, tokenrpn);

        // print the error
        if (actoperation == OP_INVALID) {
            edm::LogVerbatim("L1GlobalTriggerLogicParser") 
                << "Syntax error while parsing algorithm: " << getName() 
                << std::endl;
            errorpos = expr_iss.tellg();
            edm::LogVerbatim("L1GlobalTriggerLogicParser") 
                << expr_iss.str().substr(0,errorpos) << "<-- ERROR!" 
                << std::endl;
            edm::LogVerbatim("L1GlobalTriggerLogicParser") 
                << expr_iss.str().substr(errorpos) 
                << std::endl;
            return -1;
        }

        // The following code is written from the instructions 
        // found at http://en.wikipedia.org/wiki/Postfix_notation#Converting_from_infix_notation

        if (actoperation == OP_OPERAND) {
            // operands get pushed to the postfix notation immediately
            p_rpnvector.push_back(tokenrpn);
        }
        
        // TODO do precedence automatic and one statement for not, and, or?
        if (actoperation == OP_NOT ) {
            operatorstack.push(tokenrpn);
            // there are no operators with higher precedence
        }

        if (actoperation == OP_AND) {
            // first pop operators with higher precedence (NOT)
            while (!operatorstack.empty() && operatorstack.top().operation == OP_NOT) {
                p_rpnvector.push_back(operatorstack.top());
                operatorstack.pop();
            }            
            operatorstack.push(tokenrpn);
        }
        
        if (actoperation == OP_OR) {
            // pop operators with higher precedence (AND, NOT)
            while (!operatorstack.empty() && 
                (operatorstack.top().operation == OP_NOT || 
                 operatorstack.top().operation == OP_AND)  ) {
                
                p_rpnvector.push_back(operatorstack.top());
                operatorstack.pop();
            }
            // push operator on stack
            operatorstack.push(tokenrpn);
        }

        // left parenthesis
        if (actoperation == OP_OPENBRACKET) {
            // just push it on stack
            operatorstack.push(tokenrpn);
        }
        
        // right parenthesis
        if (actoperation == OP_CLOSEBRACKET) {

            // check if the operatorstack is empty
            if (operatorstack.empty()) {
                edm::LogVerbatim("L1GlobalTriggerLogicParser") 
                    << "Syntax error while parsing algorithm - misplaced ')': " 
                    << getName() 
                    << std::endl;
                errorpos = expr_iss.tellg();
                edm::LogVerbatim("L1GlobalTriggerLogicParser") 
                    << expr_iss.str().substr(0,errorpos) << "<-- ERROR!" 
                    << std::endl;
                edm::LogVerbatim("L1GlobalTriggerLogicParser") 
                    << expr_iss.str().substr(errorpos) 
                    << std::endl;
                return -1;
            }
            
            // pop stack until a left parenthesis is found
            do {
                if (operatorstack.top().operation != OP_OPENBRACKET) {
                    p_rpnvector.push_back(operatorstack.top()); // pop
                    operatorstack.pop();
                }
                if (operatorstack.empty()) { // the operatorstack must not be empty
                    edm::LogVerbatim("L1GlobalTriggerLogicParser") 
                        << "Syntax error while parsing algorithm - misplaced ')': " 
                        << getName() << std::endl;
                    errorpos = expr_iss.tellg();
                    edm::LogVerbatim("L1GlobalTriggerLogicParser") 
                        << expr_iss.str().substr(0,errorpos) << "<-- ERROR!" 
                        << std::endl;
                    edm::LogVerbatim("L1GlobalTriggerLogicParser") 
                        << expr_iss.str().substr(errorpos) << std::endl;
                    return -1;
                }
            } while (operatorstack.top().operation != OP_OPENBRACKET); 
                     // until the bracket is closed
            
            operatorstack.pop(); // pop the open bracket.
        }
    
        lastoperation = actoperation;    // for the next turn
    }

    // pop the rest of the operator stack

    while (!operatorstack.empty()) {
        if (operatorstack.top().operation == OP_OPENBRACKET) {
            edm::LogVerbatim("L1GlobalTriggerLogicParser") 
                << "Syntax error while parsing algorithm - missing ')': " 
                << getName() << std::endl;
            return -1;
        }
        p_rpnvector.push_back(operatorstack.top());
        operatorstack.pop();
    }

    // now count all operations and check if the result is 1
    RPNVector::iterator it;
    int counter = 0;
    for(it = p_rpnvector.begin(); it != p_rpnvector.end(); it++) {
        if (it->operation == OP_OPERAND) counter++;
        if (it->operation == OP_OR || it->operation == OP_AND) counter--;
        if (counter < 1) {
            edm::LogVerbatim("L1GlobalTriggerLogicParser") 
                << "Syntax error while parsing algorithm (too many operators) : " 
                << getName() << std::endl;
            return -1;
        }
    }

    if (counter > 1) {
        edm::LogVerbatim("L1GlobalTriggerLogicParser") 
            << "Syntax error while parsing algorithm (too many operands) : " 
            << getName() << std::endl;
        return -1;
    }

    // finaly build the postfix expression

    buildRPNExpression();
        
    return 0;
}

/**
 * setExpression Try to set an expression and build the parse tree.
 *
 * @param expression The expression to be parsed.
 *
 * @return 0 if the expression was parsed successfully. -1 if an error occured.
 *
 */

int L1GlobalTriggerLogicParser::setExpression(const std::string& expression,
    L1GlobalTriggerConfig::ConditionsMap *operandmap, unsigned int nummap) {

    std::string exprwithbs;		// expression string with spaces added before and after brackets
    
    clearRPNVector();	// clear parse tree 
    p_expression = expression;
    p_operandmap = operandmap;
    p_nummap = nummap;

    // add spaces before and after brackets
    addBracketSpaces(p_expression, exprwithbs);
    // LogDebug << exprwithbs << std::endl;
    if (buildRPNVector(exprwithbs) != 0) {
        return -1;
    }
    
    return 0;
}


/**
 * blockCondition - Check if the expression returns true. 
 *
 * @return Boolean result of the check.
 *
 */

const bool L1GlobalTriggerLogicParser::blockCondition() const {

    // we checked the expression, so it can not have stack underun. 
    // If the stack underuns we get SIGSEGV.

    // we return false if there is no expression
    if (p_rpnvector.empty()) {
        return false;	
    }
    
    stack<bool> resultstack;    // the stack that contains the temporary results
    bool b1, b2;                // temporary boolean values to do an operation

    RPNVector::const_iterator it;       
    for(it = p_rpnvector.begin(); it != p_rpnvector.end(); it++) {
        // the result of operands is simply pushed on the stack
        if (it->operation == OP_OPERAND) { 
            resultstack.push(it->operand->getLastResult());
        }
        if (it->operation == OP_NOT) {
            b1 = resultstack.top();                     // pop the top
            resultstack.pop();
            resultstack.push(!b1);                      // and push the negation
        }
        if (it->operation == OP_OR) {
            b1 = resultstack.top(); resultstack.pop();  // pop the top 2
            b2 = resultstack.top(); resultstack.pop();
            resultstack.push(b1 || b2);	                // push the result
        }
        if (it->operation == OP_AND) {
            b1 = resultstack.top(); resultstack.pop();  // pop the top 2
            b2 = resultstack.top(); resultstack.pop();
            resultstack.push(b1 && b2);                 // push the result
        }
    }

    // now we got the result in the top of the stack
    return resultstack.top();
}
    
void L1GlobalTriggerLogicParser::printThresholds() const {

    edm::LogVerbatim("L1GlobalTriggerLogicParser") 
        << "L1GlobalTriggerLogicParser: Threshold values " << std::endl;
    edm::LogVerbatim("L1GlobalTriggerLogicParser") 
        << "Condition Name:     " << getName() << std::endl;
    edm::LogVerbatim("L1GlobalTriggerLogicParser") 
        << "Output pin:         " << getOutputPin() << std::endl;
    edm::LogVerbatim("L1GlobalTriggerLogicParser") 
        << "Expression:         " << p_expression << std::endl;
    edm::LogVerbatim("L1GlobalTriggerLogicParser") 
        << "Postfix expression: " << p_rpnexpression << std::endl 
        << std::endl;
}


// rules for operations 
// the 3rd column defines what operation the operator/operand must not follow
const struct L1GlobalTriggerLogicParser::OperationRule 
    L1GlobalTriggerLogicParser::p_operationrules[] = {
        
    { "AND", OP_AND, OP_AND | OP_OR | OP_NOT | OP_OPENBRACKET | OP_NULL },
    { "OR", OP_OR, OP_AND | OP_OR | OP_NOT | OP_OPENBRACKET | OP_NULL },
    { "NOT", OP_NOT, OP_OPERAND | OP_CLOSEBRACKET | OP_NULL},
    { "(", OP_OPENBRACKET, OP_OPERAND | OP_CLOSEBRACKET },
    { ")", OP_CLOSEBRACKET, OP_AND | OP_OR | OP_NOT | OP_OPENBRACKET },
    { NULL, OP_OPERAND, OP_OPERAND | OP_CLOSEBRACKET }, // the default
    { NULL, OP_NULL, OP_NULL } 
}; 
