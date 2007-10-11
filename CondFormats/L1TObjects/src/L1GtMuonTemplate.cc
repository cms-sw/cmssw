/**
 * \class L1GtMuonTemplate
 * 
 * 
 * Description: L1 Global Trigger muon template.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date:$
 * $Revision:$
 *
 */

// this class header
#include "CondFormats/L1TObjects/interface/L1GtMuonTemplate.h"

// system include files
#include <string>

#include <iostream>
#include <iomanip>

// user include files

//   base class
#include "CondFormats/L1TObjects/interface/L1GtCondition.h"

#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

// forward declarations

// constructors
L1GtMuonTemplate::L1GtMuonTemplate()
        : L1GtCondition()
{

    m_condCategory = MuonCond;

}

L1GtMuonTemplate::L1GtMuonTemplate(const std::string& cName)
        : L1GtCondition(cName)
{

    m_condCategory = MuonCond;

}

L1GtMuonTemplate::L1GtMuonTemplate(const std::string& cName, const L1GtConditionType& cType)
        : L1GtCondition(cName, cType)
{

    m_condCategory = MuonCond;

    int nObjects = nrObjects();
    m_objectParameter.reserve(nObjects);

    m_objectType.reserve(nObjects);
    m_objectType.assign(nObjects, Mu);

}

// copy constructor
L1GtMuonTemplate::L1GtMuonTemplate(const L1GtMuonTemplate& cp)
        : L1GtCondition(cp.m_condName)
{
    copy(cp);
}

// destructor
L1GtMuonTemplate::~L1GtMuonTemplate()
{
    // empty now
}

// assign operator
L1GtMuonTemplate& L1GtMuonTemplate::operator= (const L1GtMuonTemplate& cp)
{

    copy(cp);
    return *this;
}


// setConditionParameter - set the parameters of the condition
void L1GtMuonTemplate::setConditionParameter(
    const std::vector<ObjectParameter>& objParameter,
    const CorrelationParameter& corrParameter)
{

    m_objectParameter = objParameter;
    m_correlationParameter = corrParameter;

}

void L1GtMuonTemplate::print(std::ostream& myCout) const
{

    myCout << "L1GtMuonTemplate:" << std::endl;
    myCout << "  Condition Name: " << condName() << std::endl;
    myCout << "\n  greater or equal bit: " << condGEq() << std::endl;

    int nObjects = nrObjects();

    for (int i = 0; i < nObjects; i++) {
        myCout << std::endl;
        myCout << "  Template for object " << i << std::endl;
        myCout << "    ptHighThreshold   "
        <<  std::hex << m_objectParameter[i].ptHighThreshold << std::endl;
        myCout << "    ptLowThreshold    "
        <<  std::hex << m_objectParameter[i].ptLowThreshold << std::endl;
        myCout << "    enableMip         "
        <<  m_objectParameter[i].enableMip << std::endl;
        myCout << "    enableIso         "
        <<  m_objectParameter[i].enableIso << std::endl;
        myCout << "    requestIso        "
        <<  m_objectParameter[i].requestIso << std::endl;
        myCout << "    quality           "
        <<  std::hex << m_objectParameter[i].quality << std::endl;
        myCout << "    eta               "
        <<  std::hex << m_objectParameter[i].eta << std::endl;
        myCout << "    phiHigh           "
        <<  std::hex << m_objectParameter[i].phiHigh << std::endl;
        myCout << "    phiLow            "
        <<  std::hex << m_objectParameter[i].phiLow << std::endl;
    }

    myCout << "    Correlation parameters:" <<  std::endl;

    myCout << "    chargeCorrelation    "
    << std::hex << m_correlationParameter.chargeCorrelation << std::endl;

    if ( wsc() ) {
        myCout << "    deltaEta            "
        << std::hex << m_correlationParameter.deltaEta << std::endl;
        myCout << "    deltaPhiHigh        "
        << std::hex << m_correlationParameter.deltaPhiHigh << std::endl;
        myCout << "    deltaPhiLow         "
        << std::hex << m_correlationParameter.deltaPhiLow << std::endl;
    }

    // reset to decimal output
    myCout << std::dec << std::endl;
}

void L1GtMuonTemplate::copy(const L1GtMuonTemplate& cp)
{

    m_condName     = cp.condName();
    m_condCategory = cp.condCategory();
    m_objectType   = cp.objectType();
    m_condType     = cp.condType();
    m_condGEq      = cp.condGEq();
    m_condChipNr   = cp.condChipNr();

    m_objectParameter = *(cp.objectParameter());
    m_correlationParameter = *(cp.correlationParameter());

}



