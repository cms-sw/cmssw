/**
* \class L1uGtExtBlk
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
#include "DataFormats/L1Trigger/interface/L1uGtExtBlk.h"


// system include files


// user include files

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

// constructors

// empty constructor, all members set to zero;
L1uGtExtBlk::L1uGtExtBlk(int orbitNr, int bxNr, int bxInEvent):
   m_orbitNr(orbitNr), m_bxNr(bxNr), m_bxInEvent(bxInEvent)
{

    //Clear out the header data
    m_finalOR=0;

    // Reserve/Clear out the decision words
    m_extDecision.reserve(L1GlobalTriggerReadoutSetup::NumberPhysTriggers);
    m_extDecision.assign(L1GlobalTriggerReadoutSetup::NumberPhysTriggers,false);
    
}


// empty constructor, all members set to zero;
L1uGtExtBlk::L1uGtExtBlk( )
{

    //Clear out the header data
    m_orbitNr=0;
    m_bxNr=0;
    m_bxInEvent=0;
    m_finalOR=0;

    // Reserve/Clear out the decision words
    m_extDecision.reserve(L1GlobalTriggerReadoutSetup::NumberPhysTriggers);
    m_extDecision.assign(L1GlobalTriggerReadoutSetup::NumberPhysTriggers,false);

}



// destructor
L1uGtExtBlk::~L1uGtExtBlk()
{

    // empty now
}


/// Set decision bits
void L1uGtExtBlk::setExternalDecision(int bit, bool val)   
{ 
//   if(bit < m_algoDecision.size()) {
       
      m_extDecision.at(bit) = val;   
   
//   } 
   // Need some erorr checking here.
      
}


/// Get decision bits
bool L1uGtExtBlk::getExternalDecision(unsigned int bit) const  
{ 
   if(bit>=m_extDecision.size()) return false;
   return m_extDecision.at(bit); 
}


// reset the content of a L1uGtExtBlk
void L1uGtExtBlk::reset()
{

    //Clear out the header data
    m_orbitNr=0;
    m_bxNr=0;
    m_bxInEvent=0;
    m_finalOR=0;

    // Clear out the decision words
    // but leave the vector intact 
    m_extDecision.assign(L1GlobalTriggerReadoutSetup::NumberPhysTriggers,false);


}

// pretty print the content of a L1uGtExtBlk
void L1uGtExtBlk::print(std::ostream& myCout) const
{

    myCout << " uGtExtBlk " << std::endl;
    
    myCout << "    Orbit Number (hex):  0x" << std::hex << std::setw(8) << std::setfill('0') << m_orbitNr << std::endl;

    myCout << "    Bx Number (hex):     0x" << std::hex << std::setw(4) << std::setfill('0') << m_bxNr << std::endl;

    myCout << "    Local Bx (hex):      0x" << std::hex << std::setw(1) << std::setfill('0') << m_bxInEvent << std::endl;

    // Loop through bits to create a hex word of algorithm bits.
    int lengthWd = m_extDecision.size();
    myCout << "    External Condition   0x" << std::hex;
    int digit = 0;
    for(int i=lengthWd-1; i>-1; i--) {
      if(m_extDecision.at(i)) digit |= (1 << (i%4));
      if((i%4) == 0){
         myCout << std::hex << std::setw(1) << digit;
	 digit = 0; 
      }  
    } //end loop over algorithm bits
    myCout << std::endl;    

}


