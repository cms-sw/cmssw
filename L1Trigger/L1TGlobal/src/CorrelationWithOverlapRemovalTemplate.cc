/**
 * \class CorrelationWithOverlapRemovalTemplate
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
#include "L1Trigger/L1TGlobal/interface/CorrelationWithOverlapRemovalTemplate.h"

// system include files

#include <iostream>
#include <iomanip>

// user include files
//   base class

// forward declarations

// constructors
//   default

CorrelationWithOverlapRemovalTemplate::CorrelationWithOverlapRemovalTemplate()
        : GlobalCondition()
{

    m_condCategory = l1t::CondCorrelationWithOverlapRemoval;
    m_condType = l1t::Type2corWithOverlapRemoval;
    m_condChipNr = -1;

    // there are in fact three objects
    int nObjects = nrObjects();

    if (nObjects > 0) {
        m_objectType.reserve(nObjects);
    }

    m_cond0Category = l1t::CondNull;
    m_cond1Category = l1t::CondNull;
    m_cond2Category = l1t::CondNull;
    m_cond0Index = -1;
    m_cond1Index = -1;
    m_cond2Index = -1;

}

//   from condition name
CorrelationWithOverlapRemovalTemplate::CorrelationWithOverlapRemovalTemplate(const std::string& cName)
        : GlobalCondition(cName)
{

    m_condCategory = l1t::CondCorrelationWithOverlapRemoval;
    m_condType = l1t::Type2corWithOverlapRemoval;
    m_condChipNr = -1;

    // there are in fact two objects
    int nObjects = nrObjects();

    if (nObjects > 0) {
        m_objectType.reserve(nObjects);
    }

    m_cond0Category = l1t::CondNull;
    m_cond1Category = l1t::CondNull;
    m_cond2Category = l1t::CondNull;
    m_cond0Index = -1;
    m_cond1Index = -1;
    m_cond2Index = -1;

}

//   from condition name, the category of first sub-condition, the category of the
//   second sub-condition, the index of first sub-condition in the cor* vector,
//   the index of second sub-condition in the cor* vector
CorrelationWithOverlapRemovalTemplate::CorrelationWithOverlapRemovalTemplate(const std::string& cName,
        const l1t::GtConditionCategory& cond0Cat,
        const l1t::GtConditionCategory& cond1Cat,
        const l1t::GtConditionCategory& cond2Cat,
        const int cond0Index,
        const int cond1index,
        const int cond2index) :
    GlobalCondition(cName),
            m_cond0Category(cond0Cat),
            m_cond1Category(cond1Cat),
            m_cond2Category(cond2Cat),
            m_cond0Index(cond0Index),
            m_cond1Index(cond1index),
            m_cond2Index(cond2index)

{

    m_condCategory = l1t::CondCorrelationWithOverlapRemoval;
    m_condType = l1t::Type2corWithOverlapRemoval;
    m_condChipNr = -1;

    // there are in fact two objects
    int nObjects = nrObjects();

    if (nObjects> 0) {
        m_objectType.resize(nObjects);
    }
}

// copy constructor
CorrelationWithOverlapRemovalTemplate::CorrelationWithOverlapRemovalTemplate(const CorrelationWithOverlapRemovalTemplate& cp)
        : GlobalCondition(cp.m_condName)
{
    copy(cp);
}

// destructor
CorrelationWithOverlapRemovalTemplate::~CorrelationWithOverlapRemovalTemplate()
{
    // empty now
}

// assign operator
CorrelationWithOverlapRemovalTemplate& CorrelationWithOverlapRemovalTemplate::operator= (const CorrelationWithOverlapRemovalTemplate& cp)
{

    copy(cp);
    return *this;
}

// set the category of the two sub-conditions
void CorrelationWithOverlapRemovalTemplate::setCond0Category(
        const l1t::GtConditionCategory& condCateg) {

    m_cond0Category = condCateg;
}

void CorrelationWithOverlapRemovalTemplate::setCond1Category(
        const l1t::GtConditionCategory& condCateg) {

    m_cond1Category = condCateg;
}

void CorrelationWithOverlapRemovalTemplate::setCond2Category(
        const l1t::GtConditionCategory& condCateg) {

    m_cond2Category = condCateg;
}


// set the index of the two sub-conditions in the cor* vector from menu
void CorrelationWithOverlapRemovalTemplate::setCond0Index(const int& condIndex) {
    m_cond0Index = condIndex;
}

void CorrelationWithOverlapRemovalTemplate::setCond1Index(const int& condIndex) {
    m_cond1Index = condIndex;
}

void CorrelationWithOverlapRemovalTemplate::setCond2Index(const int& condIndex) {
    m_cond2Index = condIndex;
}


// set the correlation parameters of the condition
void CorrelationWithOverlapRemovalTemplate::setCorrelationWithOverlapRemovalParameter(
        const CorrelationWithOverlapRemovalParameter& corrParameter) {

    m_correlationParameter = corrParameter;

}

void CorrelationWithOverlapRemovalTemplate::print(std::ostream& myCout) const
{

    myCout << "\n  CorrelationWithOverlapRemovalTemplate print..." << std::endl;

    GlobalCondition::print(myCout);

    myCout << "\n  First sub-condition category:  " << m_cond0Category <<  std::endl;
    myCout <<   "  Second sub-condition category: " << m_cond1Category <<  std::endl;
    myCout <<   "  Third sub-condition category: " << m_cond2Category <<  std::endl;

    myCout << "\n  First sub-condition index:  " << m_cond0Index <<  std::endl;
    myCout <<   "  Second sub-condition index: " << m_cond1Index <<  std::endl;
    myCout <<   "  Third sub-condition index: " << m_cond2Index <<  std::endl;

    myCout << "\n  CorrelationWithOverlapRemoval parameters " << "[ hex ]" <<  std::endl;

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
    myCout << "    minOverlapRemovalEtaCutValue        = " << std::dec << m_correlationParameter.minOverlapRemovalEtaCutValue << std::endl;
    myCout << "    maxOverlapRemovalEtaCutValue        = " << std::dec << m_correlationParameter.maxOverlapRemovalEtaCutValue << std::endl;
    myCout << "    precOverlapRemovalEtaCut            = " << std::dec << m_correlationParameter.precOverlapRemovalEtaCut     << std::endl;
    myCout << "    minOverlapRemovalPhiCutValue        = " << std::dec << m_correlationParameter.minOverlapRemovalPhiCutValue << std::endl;
    myCout << "    maxOverlapRemovalPhiCutValue        = " << std::dec << m_correlationParameter.maxOverlapRemovalPhiCutValue << std::endl;
    myCout << "    precOverlapRemovalPhiCut            = " << std::dec << m_correlationParameter.precOverlapRemovalPhiCut     << std::endl;
    myCout << "    minOverlapRemovalDRCutValue         = " << std::dec << m_correlationParameter.minOverlapRemovalDRCutValue  << std::endl;
    myCout << "    maxOverlapRemovalDRCutValue         = " << std::dec << m_correlationParameter.maxOverlapRemovalDRCutValue  << std::endl;
    myCout << "    precOverlapRemovalDRCut             = " << std::dec << m_correlationParameter.precOverlapRemovalDRCut      << std::endl;
 
    myCout << "    chargeCorrelation  = " << std::dec << m_correlationParameter.chargeCorrelation << std::endl;

    // reset to decimal output
    myCout << std::dec << std::endl;
}

void CorrelationWithOverlapRemovalTemplate::copy(const CorrelationWithOverlapRemovalTemplate& cp)
{

    m_condName     = cp.condName();
    m_condCategory = cp.condCategory();
    m_condType     = cp.condType();
    m_objectType   = cp.objectType();
    m_condGEq      = cp.condGEq();
    m_condChipNr   = cp.condChipNr();

    m_cond0Category = cp.cond0Category();
    m_cond1Category = cp.cond1Category();
    m_cond2Category = cp.cond2Category();
    m_cond0Index = cp.cond0Index();
    m_cond1Index = cp.cond1Index();
    m_cond2Index = cp.cond2Index();

    m_correlationParameter = *(cp.correlationParameter());

}

// output stream operator
std::ostream& operator<<(std::ostream& os, const CorrelationWithOverlapRemovalTemplate& result)
{
    result.print(os);
    return os;

}



