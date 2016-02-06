////
/// \class l1t::L1TGlobalScales.cc
///
/// Description: Dump Accessors for L1 GT Result.
///
/// Implementation:
///    
///
/// \author: Brian Winer Ohio State
///
/// 
#include "L1Trigger/L1TGlobal/interface/L1TGlobalScales.h"

#include <iostream>
#include <fstream>
#include <iomanip>



#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"


// constructor
l1t::L1TGlobalScales::L1TGlobalScales() 
{

 
}

// destructor
l1t::L1TGlobalScales::~L1TGlobalScales() {
 
}

std::string l1t::L1TGlobalScales::getScalesName() const { return m_ScaleSetName;}

void l1t::L1TGlobalScales::print(std::ostream& myCout) const
{

    myCout << "\n  *************  L1T Global Scales ************" << std::endl;
 
 
    myCout << "   Muon Scales: " << std::endl;
    printScale(m_muScales,myCout);
       	       
    myCout << "   EG Scales: "<< std::endl;
    printScale(m_egScales,myCout);	    

    myCout << "   Tau Scales: "<< std::endl;
    printScale(m_tauScales,myCout);
	   
    myCout << "   Jet Scales: "<< std::endl;
    printScale(m_jetScales,myCout);
	   
	   
    myCout << "   HTT Scales: "<< std::endl;
    printScale(m_httScales,myCout);
	   
    myCout << "   ETT Scales: "<< std::endl;
    printScale(m_ettScales,myCout);
	   
    myCout << "   HTM Scales: "<< std::endl;
    printScale(m_htmScales,myCout);
	   
    myCout << "   ETM Scales: "<< std::endl;
    printScale(m_etmScales,myCout);	   	   	   	   	   
	   
}
void l1t::L1TGlobalScales::printScale(ScaleParameters scale, std::ostream& myCout) const
{

    myCout <<   "    Pt Min   = "  << std::setw(10) << scale.etMin
	   << "      Pt Max   = "  << std::setw(10) << scale.etMax
	   << "      Pt Step  = "  << std::setw(10) << scale.etStep
           << "\n    Phi Min  = "  << std::setw(10) << scale.phiMin
	   << "      Phi Max  = "  << std::setw(10) << scale.phiMax
	   << "      Phi Step = "  << std::setw(10) << scale.phiStep	    
           << "\n    Eta Min  = "  << std::setw(10) << scale.etaMin
	   << "      Eta Max  = "  << std::setw(10) << scale.etaMax
	   << "      Eta Step = "  << std::setw(10) << scale.etaStep 
           << std::endl;

}
