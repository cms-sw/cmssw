/**
 * \class L1GtCorrelationTemplate
 * 
 * 
 * Description: L1 Global Trigger correlation template.  
 * Includes spatial correlation for two objects of different type.
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

// this class header
#include "CondFormats/L1TObjects/interface/L1GtCorrelationTemplate.h"

// system include files

#include <iostream>
#include <iomanip>

// user include files

//   base class

#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

// forward declarations

// constructors
//   default

L1GtCorrelationTemplate::L1GtCorrelationTemplate()
        : L1GtCondition()
{

    m_condCategory = CondCorrelation;
    m_condType = Type2cor;
    m_condChipNr = -1;

    // there are in fact two objects
    int nObjects = nrObjects();

    if (nObjects > 0) {
        m_objectCondition.reserve(nObjects);
        m_objectType.reserve(nObjects);
    }


}

//   from condition name
L1GtCorrelationTemplate::L1GtCorrelationTemplate(const std::string& cName)
        : L1GtCondition(cName)
{

    m_condCategory = CondCorrelation;
    m_condType = Type2cor;
    m_condChipNr = -1;

    // there are in fact two objects
    int nObjects = nrObjects();

    if (nObjects > 0) {
        m_objectCondition.reserve(nObjects);
        m_objectType.reserve(nObjects);
    }

}

//   from condition name and two existing conditions
L1GtCorrelationTemplate::L1GtCorrelationTemplate(const std::string& cName,
        const L1GtCondition& cond0, const L1GtCondition& cond1)
        : L1GtCondition(cName)
{

    m_condCategory = CondCorrelation;
    m_condType = Type2cor;
    m_condChipNr = -1;


    // check that both conditions are of correct types
    // (two objects of different type, with spatial correlations)
    // TODO FIXME

    // there are in fact two objects
    int nObjects = nrObjects();

    if (nObjects > 0) {
        m_objectCondition.reserve(nObjects);
        m_objectType.resize(nObjects);

        std::vector<L1GtObject> objVec = cond0.objectType();
        m_objectType[0] = objVec[0];

        objVec = cond1.objectType();
        m_objectType[1] = objVec[1];

    }

}

// copy constructor
L1GtCorrelationTemplate::L1GtCorrelationTemplate(const L1GtCorrelationTemplate& cp)
        : L1GtCondition(cp.m_condName)
{
    copy(cp);
}

// destructor
L1GtCorrelationTemplate::~L1GtCorrelationTemplate()
{
    // empty now
}

// assign operator
L1GtCorrelationTemplate& L1GtCorrelationTemplate::operator= (const L1GtCorrelationTemplate& cp)
{

    copy(cp);
    return *this;
}


// setConditionParameter - set the parameters of the condition
void L1GtCorrelationTemplate::setConditionParameter(
    const std::vector<L1GtCondition>& objCondition,
    const CorrelationParameter& corrParameter)
{

    m_objectCondition = objCondition;
    m_correlationParameter = corrParameter;

}

void L1GtCorrelationTemplate::print(std::ostream& myCout) const
{

    myCout << "\n  L1GtCorrelationTemplate print..." << std::endl;

    L1GtCondition::print(myCout);

    int nObjects = nrObjects();

    for (int i = 0; i < nObjects; i++) {

        m_objectCondition[i].print(myCout);

    }

    myCout << "  Correlation parameters " << "[ hex ]" <<  std::endl;


    myCout << "    deltaEtaRange      = "
    << std::hex << m_correlationParameter.deltaEtaRange << std::endl;
    myCout << "    deltaPhiRange      = "
    << std::hex << m_correlationParameter.deltaPhiRange << std::endl;
    myCout << "    deltaPhiMaxbits    = "
    << std::hex << m_correlationParameter.deltaPhiMaxbits << std::endl;

    // reset to decimal output
    myCout << std::dec << std::endl;
    myCout << "\n  ...end L1GtCorrelationTemplate print." << std::endl;
}

void L1GtCorrelationTemplate::copy(const L1GtCorrelationTemplate& cp)
{

    m_condName     = cp.condName();
    m_condCategory = cp.condCategory();
    m_condType     = cp.condType();
    m_objectType   = cp.objectType();
    m_condGEq      = cp.condGEq();
    m_condChipNr   = cp.condChipNr();

    m_objectCondition = *(cp.objectCondition());
    m_correlationParameter = *(cp.correlationParameter());

}



