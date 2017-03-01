#ifndef L1GlobalTrigger_L1TGtObjectMap_h
#define L1GlobalTrigger_L1TGtObjectMap_h

/**
 * \class GlobalObjectMap
 * 
 * 
 * Description: map trigger objects to an algorithm and the conditions therein.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 *
 */

// system include files
#include <string>
#include <vector>

#include <iosfwd>

// user include files
#include "DataFormats/L1TGlobal/interface/GlobalObjectMapFwd.h"

#include "DataFormats/L1TGlobal/interface/GlobalLogicParser.h"

// forward declarations

// class declaration
class GlobalObjectMap
{

public:

    /// constructor(s)
  GlobalObjectMap(){}

    /// destructor
  //~GlobalObjectMap(){}

public:

    /// get / set name for algorithm in the object map
    inline const std::string & algoName() const
    {
        return m_algoName;
    }

    void setAlgoName(const std::string& algoNameValue) {
        m_algoName = algoNameValue;
    }

    /// get / set bit number for algorithm in the object map
    inline int algoBitNumber() const
    {
        return m_algoBitNumber;
    }

    void setAlgoBitNumber(int algoBitNumberValue)
    {
        m_algoBitNumber = algoBitNumberValue;
    }

    /// get / set the GTL result for algorithm
    /// NOTE: FDL can mask an algorithm!
    inline bool algoGtlResult() const {
        return m_algoGtlResult;
    }

    void setAlgoGtlResult(bool algoGtlResultValue) {
        m_algoGtlResult = algoGtlResultValue;
    }

    /// get / set the vector of combinations for the algorithm
    /// return a constant reference to the vector of combinations for the algorithm
    inline const std::vector<CombinationsInCond>& combinationVector() const
    {
        return m_combinationVector;
    }

    void setCombinationVector(const std::vector<CombinationsInCond>& combinationVectorValue) {
        m_combinationVector = combinationVectorValue;
    }
    void swapCombinationVector(std::vector<CombinationsInCond>& combinationVectorValue) {
      m_combinationVector.swap(combinationVectorValue);
    }

    /// get / set the vector of operand tokens
    /// return a constant reference to the vector of operand tokens
    inline const std::vector<GlobalLogicParser::OperandToken>& operandTokenVector() const {
        return m_operandTokenVector;
    }
    
    void setOperandTokenVector(const std::vector<GlobalLogicParser::OperandToken>& operandTokenVectorValue) {
        m_operandTokenVector = operandTokenVectorValue;
    }
    void swapOperandTokenVector(std::vector<GlobalLogicParser::OperandToken>& operandTokenVectorValue) {
      m_operandTokenVector.swap(operandTokenVectorValue);
    }


    /// get / set the vector of object types
    /// return a constant reference to the vector of operand tokens
    inline const std::vector<ObjectTypeInCond>& objectTypeVector() const {
        return m_objectTypeVector;
    }
    void setObjectTypeVector(const std::vector<ObjectTypeInCond>& objectTypeVectorValue) {
        m_objectTypeVector = objectTypeVectorValue;
    }
    void swapObjectTypeVector(std::vector<ObjectTypeInCond>& objectTypeVectorValue) {
      m_objectTypeVector.swap(objectTypeVectorValue);
    }
    
public:

    /// return all the combinations passing the requirements imposed in condition condNameVal
    const CombinationsInCond* getCombinationsInCond(const std::string& condNameVal) const;

    /// return all the combinations passing the requirements imposed in condition condNumberVal
    const CombinationsInCond* getCombinationsInCond(const int condNumberVal) const;

    /// return the result for the condition condNameVal
    const bool getConditionResult(const std::string& condNameVal) const;

public:

    /// reset the object map 
    void reset();
    
    /// print the full object map
    void print(std::ostream& myCout) const;
    
private:

    // name of the algorithm
    std::string m_algoName;

    // bit number for algorithm
    int m_algoBitNumber;

    // GTL result of the algorithm
    bool m_algoGtlResult;

    /// vector of operand tokens for an algorithm 
    /// (condition name, condition index, condition result)
    std::vector<GlobalLogicParser::OperandToken> m_operandTokenVector;
    
    // vector of combinations for all conditions in an algorithm
    std::vector<CombinationsInCond> m_combinationVector;

    // vector of object type vectors for all conditions in an algorithm
    std::vector<ObjectTypeInCond> m_objectTypeVector;

};

#endif /* L1GlobalTrigger_L1TGtObjectMap_h */
