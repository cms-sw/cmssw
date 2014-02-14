/**
* \class L1uGtRecBlk
*
*
* Description: see header file.
*
* Implementation:
* <TODO: enter implementation details>
*
* \author: Brian Winer -- Ohio State
*
*
*/

// this class header
#include "DataFormats/L1Trigger/interface/L1uGtRecBlk.h"


// system include files


// user include files

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

// constructors

// empty constructor, all members set to zero;
L1uGtRecBlk::L1uGtRecBlk(int ver, int bxAlg, int bxExt, int bxMu, int bxCal, int psIndex,
                         cms_uint64_t orb, int bxNr, int lumiS, int uGtNr):
                        m_firmVersion(ver),	      
                        m_totBxInEvent_alg(bxAlg),    
                        m_totBxInEvent_ext(bxExt),    
                        m_totBxInEvent_muData(bxMu), 
                        m_totBxInEvent_calData(bxCal),
                        m_prescaleIndex(psIndex),
			m_orbitNr(orb),  
			m_bxNr(bxNr),	
			m_lumiSection(lumiS), 
			m_internalEvt(uGtNr) 
{

     // Other Quantities Empty
     m_finalOR.clear();
     m_triggerNr = 0;

  
}


// empty constructor, all members set to zero;
L1uGtRecBlk::L1uGtRecBlk( )
{

      m_firmVersion = 0;	    
      m_totBxInEvent_alg = 0;    
      m_totBxInEvent_ext = 0;    
      m_totBxInEvent_muData = 0; 
      m_totBxInEvent_calData = 0;
      m_prescaleIndex = 0;
      m_orbitNr = 0;  
      m_bxNr = 0;   
      m_lumiSection = 0; 
      m_internalEvt = 0; 
      m_finalOR.clear();
      m_triggerNr = 0;

}



// destructor
L1uGtRecBlk::~L1uGtRecBlk()
{

    // empty now
}




// reset the content of a L1uGtRecBlk
void L1uGtRecBlk::reset()
{

      m_firmVersion = 0;	    
      m_totBxInEvent_alg = 0;    
      m_totBxInEvent_ext = 0;    
      m_totBxInEvent_muData = 0; 
      m_totBxInEvent_calData = 0;
      m_prescaleIndex = 0;
      m_orbitNr = 0;  
      m_bxNr = 0;   
      m_lumiSection = 0; 
      m_internalEvt = 0; 
      m_finalOR.clear();
      m_triggerNr = 0;


}

// pretty print the content of a L1uGtRecBlk
void L1uGtRecBlk::print(std::ostream& myCout) const
{

    
    myCout << " uGtRecBlk " << std::endl;

    myCout << "    Firmware Version (hex):         0x" << std::hex << std::setw(16) << std::setfill('0') << m_firmVersion << std::endl;

    myCout << "    bx in Events (alg,ext,mu,Cal):  "  
           << m_totBxInEvent_alg << ",   "
	   << m_totBxInEvent_ext << ",   "
	   << m_totBxInEvent_muData << ",   "
	   << m_totBxInEvent_calData << ",   "
	   << std::endl;
     
    myCout << "    PreScale Index (hex):           0x" << std::hex << std::setw(2) << std::setfill('0') << m_prescaleIndex << std::endl;

    myCout << "    Trigger Number (hex):           0x" << std::hex << std::setw(16) << std::setfill('0') << m_triggerNr << std::endl;
    
    myCout << "    Orbit Number (hex):             0x" << std::hex << std::setw(12) << std::setfill('0') << m_orbitNr << std::endl;

    myCout << "    Bx Number (hex):                0x" << std::hex << std::setw(4) << std::setfill('0') << m_bxNr << std::endl;

    myCout << "    Lumi Section (hex):             0x" << std::hex << std::setw(8) << std::setfill('0') << m_lumiSection << std::endl;

    myCout << "    Internal Evt (hex):             0x" << std::hex << std::setw(8) << std::setfill('0') << m_internalEvt << std::endl;
     
}


