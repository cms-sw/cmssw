/**
 * \class MuonTemplate
 *
 *
 * Description: L1 Global Trigger muon template.
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
#include "L1Trigger/L1TGlobal/interface/MuonTemplate.h"

// system include files

#include <iostream>
#include <iomanip>

// user include files

//   base class

#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

// forward declarations

// constructors
MuonTemplate::MuonTemplate()
        : GtCondition()
{

    m_condCategory = l1t::CondMuon;

}

MuonTemplate::MuonTemplate(const std::string& cName)
        : GtCondition(cName)
{

    m_condCategory = l1t::CondMuon;

}

MuonTemplate::MuonTemplate(const std::string& cName, const l1t::GtConditionType& cType)
        : GtCondition(cName, l1t::CondMuon, cType)
{

    int nObjects = nrObjects();

    if (nObjects > 0) {
        m_objectParameter.reserve(nObjects);

        m_objectType.reserve(nObjects);
        m_objectType.assign(nObjects, Mu);
    }

}

// copy constructor
MuonTemplate::MuonTemplate(const MuonTemplate& cp)
        : GtCondition(cp.m_condName)
{
    copy(cp);
}

// destructor
MuonTemplate::~MuonTemplate()
{
    // empty now
}

// assign operator
MuonTemplate& MuonTemplate::operator= (const MuonTemplate& cp)
{

    copy(cp);
    return *this;
}


// setConditionParameter - set the parameters of the condition
void MuonTemplate::setConditionParameter(
    const std::vector<ObjectParameter>& objParameter,
    const CorrelationParameter& corrParameter)
{

    m_objectParameter = objParameter;
    m_correlationParameter = corrParameter;

}

void MuonTemplate::print(std::ostream& myCout) const
{

    myCout << "\n  MuonTemplate print..." << std::endl;

    GtCondition::print(myCout);

    int nObjects = nrObjects();

    for (int i = 0; i < nObjects; i++) {
        myCout << std::endl;
        myCout << "  Template for object " << i << " [ hex ]" << std::endl;
        myCout << "    ptHighThreshold   = "
        << std::hex << m_objectParameter[i].ptHighThreshold << std::endl;
        myCout << "    ptLowThreshold    = "
        << std::hex << m_objectParameter[i].ptLowThreshold << std::endl;
        myCout << "    enableMip         = "
        << std::hex << m_objectParameter[i].enableMip << std::endl;
        myCout << "    enableIso         = "
        << std::hex << m_objectParameter[i].enableIso << std::endl;
        myCout << "    requestIso        = "
        << std::hex << m_objectParameter[i].requestIso << std::endl;
        myCout << "    qualityLUT        = "
        << std::hex << m_objectParameter[i].qualityLUT << std::endl;
        myCout << "    isolationLUT      = "
        << std::hex << m_objectParameter[i].isolationLUT << std::endl;
        myCout << "    etaRange          = "
        << std::hex << m_objectParameter[i].etaRange << std::endl;
        myCout << "    phiHigh           = "
        << std::hex << m_objectParameter[i].phiHigh << std::endl;
        myCout << "    phiLow            = "
        << std::hex << m_objectParameter[i].phiLow << std::endl;
    }


    if ( wsc() ) {
        myCout << "  Correlation parameters " << "[ hex ]" <<  std::endl;

        myCout <<     "    chargeCorrelation  = "
        << std::hex << m_correlationParameter.chargeCorrelation
        << std::endl;

        myCout << "    deltaEtaRange      = "
        << std::hex << m_correlationParameter.deltaEtaRange << std::endl;
        myCout << "    deltaPhiRange1Word = "
        << std::hex << m_correlationParameter.deltaPhiRange1Word << std::endl;
        myCout << "    deltaPhiRange0Word = "
        << std::hex << m_correlationParameter.deltaPhiRange0Word << std::endl;
        myCout << "    deltaPhiMaxbits    = "
        << std::hex << m_correlationParameter.deltaPhiMaxbits << std::endl;
    } else {

        if (m_condType == l1t::Type1s) {
            myCout << "  Correlation parameters " << "[ hex ]" <<  std::endl;

            myCout <<     "    chargeCorrelation  = "
            << std::hex << m_correlationParameter.chargeCorrelation
            << " (charge sign) " << std::endl;

        } else {

            myCout << "\n  Correlation parameters " << "[ hex ]" <<  std::endl;

            myCout <<     "    chargeCorrelation  = "
            << std::hex << m_correlationParameter.chargeCorrelation
            << std::endl;
        }
    }

    // reset to decimal output
    myCout << std::dec << std::endl;
}

void MuonTemplate::copy(const MuonTemplate& cp)
{

    m_condName     = cp.condName();
    m_condCategory = cp.condCategory();
    m_condType     = cp.condType();
    m_objectType   = cp.objectType();
    m_condGEq      = cp.condGEq();
    m_condChipNr   = cp.condChipNr();
    m_condRelativeBx = cp.condRelativeBx();

    m_objectParameter = *(cp.objectParameter());
    m_correlationParameter = *(cp.correlationParameter());

}

// output stream operator
std::ostream& operator<<(std::ostream& os, const MuonTemplate& result)
{
    result.print(os);
    return os;

}


