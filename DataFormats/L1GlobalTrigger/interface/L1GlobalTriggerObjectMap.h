#ifndef L1GlobalTrigger_L1GlobalTriggerObjectMap_h
#define L1GlobalTrigger_L1GlobalTriggerObjectMap_h

/**
 * \class L1GlobalTriggerObjectMap
 * 
 * 
 * Description: map trigger objects to an algorithm and the conditions therein.  
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

#include <iosfwd>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

// forward declarations

// class declaration
class L1GlobalTriggerObjectMap
{

public:

    /// constructor(s)
    L1GlobalTriggerObjectMap();

    /// destructor
    virtual ~L1GlobalTriggerObjectMap();

public:

    /// get / set name for algorithm in the object map
    inline const std::string algoName() const
    {
        return m_algoName;
    }

    void setAlgoName(std::string algoNameValue)
    {
        m_algoName = algoNameValue;
    }

    /// get / set bit number for algorithm in the object map
    inline const int algoBitNumber() const
    {
        return m_algoBitNumber;
    }

    void setAlgoBitNumber(int algoBitNumberValue)
    {
        m_algoBitNumber = algoBitNumberValue;
    }

    /// get / set the GTL result for algorithm
    /// NOTE: FDL can mask an algorithm!
    inline const bool algoGtlResult() const
    {
        return m_algoGtlResult;
    }

    void setAlgoGtlResult(bool algoGtlResultValue)
    {
        m_algoGtlResult = algoGtlResultValue;
    }

    /// get / set logical expression for algorithm in the object map
    inline const std::string algoLogicalExpression() const
    {
        return m_algoLogicalExpression;
    }

    void setAlgoLogicalExpression(std::string algoLogicalExpressionValue)
    {
        m_algoLogicalExpression = algoLogicalExpressionValue;
    }

    /// get / set numerical expression for algorithm in the object map
    inline const std::string algoNumericalExpression() const
    {
        return m_algoNumericalExpression;
    }

    void setAlgoNumericalExpression(std::string algoNumericalExpressionValue)
    {
        m_algoNumericalExpression = algoNumericalExpressionValue;
    }

    /// get / set the vector of combinations for the algorithm
    inline const std::vector<CombinationsInCond>& combinationVector() const
    {
        return m_combinationVector;
    }

    void setCombinationVector(std::vector<CombinationsInCond> combinationVectorValue)
    {
        m_combinationVector = combinationVectorValue;
    }

    /// get / set the vector of object types for the algorithm
    inline const std::vector<ObjectTypeInCond>& objectTypeVector() const
    {
        return m_objectTypeVector;
    }

    void setObjectTypeVector(std::vector<ObjectTypeInCond> objectTypeVectorValue)
    {
        m_objectTypeVector = objectTypeVectorValue;
    }

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

    // logical expression for the algorithm
    std::string m_algoLogicalExpression;

    // numerical expression for the algorithm
    // (logical expression with conditions replaced with the actual values)
    std::string m_algoNumericalExpression;

    // vector of combinations for all conditions in an algorithm
    std::vector<CombinationsInCond> m_combinationVector;

    // vector of object types for all conditions in an algorithm
    std::vector<ObjectTypeInCond> m_objectTypeVector;

};

#endif /* L1GlobalTrigger_L1GlobalTriggerObjectMap_h */
