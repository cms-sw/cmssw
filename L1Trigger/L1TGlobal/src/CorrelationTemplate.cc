/**
 * \class CorrelationTemplate
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
#include "L1Trigger/L1TGlobal/interface/CorrelationTemplate.h"

// system include files

#include <iostream>
#include <iomanip>

// user include files
//   base class

// forward declarations

// constructors
//   default

CorrelationTemplate::CorrelationTemplate()
        : GlobalCondition()
{

    m_condCategory = l1t::CondCorrelation;
    m_condType = l1t::Type2cor;
    m_condChipNr = -1;

    // there are in fact two objects
    int nObjects = nrObjects();

    if (nObjects > 0) {
        m_objectType.reserve(nObjects);
    }

    m_cond0Category = l1t::CondNull;
    m_cond1Category = l1t::CondNull;
    m_cond0Index = -1;
    m_cond1Index = -1;

}

//   from condition name
CorrelationTemplate::CorrelationTemplate(const std::string& cName)
        : GlobalCondition(cName)
{

    m_condCategory = l1t::CondCorrelation;
    m_condType = l1t::Type2cor;
    m_condChipNr = -1;

    // there are in fact two objects
    int nObjects = nrObjects();

    if (nObjects > 0) {
        m_objectType.reserve(nObjects);
    }

    m_cond0Category = l1t::CondNull;
    m_cond1Category = l1t::CondNull;
    m_cond0Index = -1;
    m_cond1Index = -1;

}

//   from condition name, the category of first sub-condition, the category of the
//   second sub-condition, the index of first sub-condition in the cor* vector,
//   the index of second sub-condition in the cor* vector
CorrelationTemplate::CorrelationTemplate(const std::string& cName,
        const l1t::GtConditionCategory& cond0Cat,
        const l1t::GtConditionCategory& cond1Cat,
        const int cond0Index,
        const int cond1index) :
    GlobalCondition(cName),
            m_cond0Category(cond0Cat),
            m_cond1Category(cond1Cat),
            m_cond0Index(cond0Index),
            m_cond1Index(cond1index)

{

    m_condCategory = l1t::CondCorrelation;
    m_condType = l1t::Type2cor;
    m_condChipNr = -1;

    // there are in fact two objects
    int nObjects = nrObjects();

    if (nObjects> 0) {
        m_objectType.resize(nObjects);
    }
}

// copy constructor
CorrelationTemplate::CorrelationTemplate(const CorrelationTemplate& cp)
        : GlobalCondition(cp.m_condName)
{
    copy(cp);
}

// destructor
CorrelationTemplate::~CorrelationTemplate()
{
    // empty now
}

// assign operator
CorrelationTemplate& CorrelationTemplate::operator= (const CorrelationTemplate& cp)
{

    copy(cp);
    return *this;
}

// set the category of the two sub-conditions
void CorrelationTemplate::setCond0Category(
        const l1t::GtConditionCategory& condCateg) {

    m_cond0Category = condCateg;
}

void CorrelationTemplate::setCond1Category(
        const l1t::GtConditionCategory& condCateg) {

    m_cond1Category = condCateg;
}


// set the index of the two sub-conditions in the cor* vector from menu
void CorrelationTemplate::setCond0Index(const int& condIndex) {
    m_cond0Index = condIndex;
}

void CorrelationTemplate::setCond1Index(const int& condIndex) {
    m_cond1Index = condIndex;
}


// set the correlation parameters of the condition
void CorrelationTemplate::setCorrelationParameter(
        const CorrelationParameter& corrParameter) {

    m_correlationParameter = corrParameter;

}

void CorrelationTemplate::print(std::ostream& myCout) const
{

    myCout << "\n  CorrelationTemplate print..." << std::endl;

    GlobalCondition::print(myCout);

    myCout << "\n  First sub-condition category:  " << m_cond0Category <<  std::endl;
    myCout <<   "  Second sub-condition category: " << m_cond1Category <<  std::endl;

    myCout << "\n  First sub-condition index:  " << m_cond0Index <<  std::endl;
    myCout <<   "  Second sub-condition index: " << m_cond1Index <<  std::endl;

    myCout << "\n  Correlation parameters " << "[ hex ]" <<  std::endl;

    myCout << "    Cut Type:  " << m_correlationParameter.corrCutType << std::endl;
    myCout << "    minEtaCutValue        = " << std::dec << m_correlationParameter.minEtaCutValue << std::endl;
    myCout << "    maxEtaCutValue        = " << std::dec << m_correlationParameter.maxEtaCutValue << std::endl;
    myCout << "    precEtaCut            = " << std::dec << m_correlationParameter.precEtaCut     << std::endl;
    myCout << "    minPhiCutValue        = " << std::dec << m_correlationParameter.minPhiCutValue << std::endl;
    myCout << "    maxPhiCutValue        = " << std::dec << m_correlationParameter.maxPhiCutValue << std::endl;
    myCout << "    precPhiCut            = " << std::dec << m_correlationParameter.precPhiCut     << std::endl;
    myCout << "    minDRCutValue         = " << std::dec << m_correlationParameter.minDRCutValue  << std::endl;
    myCout << "    maxDRCutValue         = " << std::dec << m_correlationParameter.maxDRCutValue  << std::endl;
    myCout << "    precDRCut             = " << std::dec << m_correlationParameter.precDRCut      << std::endl;
    myCout << "    minMassCutValue       = " << std::dec << m_correlationParameter.minMassCutValue<< std::endl;
    myCout << "    maxMassCutValue       = " << std::dec << m_correlationParameter.maxMassCutValue<< std::endl;
    myCout << "    precMassCut           = " << std::dec << m_correlationParameter.precMassCut    << std::endl;
 
    myCout << "    chargeCorrelation  = " << std::dec << m_correlationParameter.chargeCorrelation << std::endl;

    // reset to decimal output
    myCout << std::dec << std::endl;
}

void CorrelationTemplate::copy(const CorrelationTemplate& cp)
{

    m_condName     = cp.condName();
    m_condCategory = cp.condCategory();
    m_condType     = cp.condType();
    m_objectType   = cp.objectType();
    m_condGEq      = cp.condGEq();
    m_condChipNr   = cp.condChipNr();

    m_cond0Category = cp.cond0Category();
    m_cond1Category = cp.cond1Category();
    m_cond0Index = cp.cond0Index();
    m_cond1Index = cp.cond1Index();

    m_correlationParameter = *(cp.correlationParameter());

}

// output stream operator
std::ostream& operator<<(std::ostream& os, const CorrelationTemplate& result)
{
    result.print(os);
    return os;

}



