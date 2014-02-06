/**
* \class L1uGtAlgBlk
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
#include "DataFormats/L1Trigger/interface/L1uGtAlgBlk.h"


// system include files


// user include files

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

// constructors

// empty constructor, all members set to zero;
L1uGtAlgBlk::L1uGtAlgBlk(int orbitNr, int bxNr, int bxInEvent):
   m_orbitNr(orbitNr), m_bxNr(bxNr), m_bxInEvent(bxInEvent)
{

    //Clear out the header data
    m_finalOR=0;

    // Reserve/Clear out the decision words
    m_algoDecisionInitial.reserve(L1GlobalTriggerReadoutSetup::NumberPhysTriggers);
    m_algoDecisionInitial.assign(L1GlobalTriggerReadoutSetup::NumberPhysTriggers,false);
    
    m_algoDecisionPreScaled.reserve(L1GlobalTriggerReadoutSetup::NumberPhysTriggers);
    m_algoDecisionPreScaled.assign(L1GlobalTriggerReadoutSetup::NumberPhysTriggers,false);

    m_algoDecisionFinal.reserve(L1GlobalTriggerReadoutSetup::NumberPhysTriggers);
    m_algoDecisionFinal.assign(L1GlobalTriggerReadoutSetup::NumberPhysTriggers,false);

}


// empty constructor, all members set to zero;
L1uGtAlgBlk::L1uGtAlgBlk( )
{

    //Clear out the header data
    m_orbitNr=0;
    m_bxNr=0;
    m_bxInEvent=0;
    m_finalOR=0;

    // Reserve/Clear out the decision words
    m_algoDecisionInitial.reserve(L1GlobalTriggerReadoutSetup::NumberPhysTriggers);
    m_algoDecisionInitial.assign(L1GlobalTriggerReadoutSetup::NumberPhysTriggers,false);
    
    m_algoDecisionPreScaled.reserve(L1GlobalTriggerReadoutSetup::NumberPhysTriggers);
    m_algoDecisionPreScaled.assign(L1GlobalTriggerReadoutSetup::NumberPhysTriggers,false);

    m_algoDecisionFinal.reserve(L1GlobalTriggerReadoutSetup::NumberPhysTriggers);
    m_algoDecisionFinal.assign(L1GlobalTriggerReadoutSetup::NumberPhysTriggers,false);

}



// destructor
L1uGtAlgBlk::~L1uGtAlgBlk()
{

    // empty now
}


/// Set decision bits
void L1uGtAlgBlk::setAlgoDecisionInitial(int bit, bool val)   
{ 
//   if(bit < m_algoDecisionInitial.size()) {
       
      m_algoDecisionInitial.at(bit) = val;   
   
 //  } 
   // Need some erorr checking here.
   
   
}
void L1uGtAlgBlk::setAlgoDecisionPreScaled(int bit, bool val) 
{ 
   m_algoDecisionPreScaled.at(bit) = val; 
}
void L1uGtAlgBlk::setAlgoDecisionFinal(int bit, bool val)     
{ 
   m_algoDecisionFinal.at(bit) = val; 
}

/// Get decision bits
bool L1uGtAlgBlk::getAlgoDecisionInitial(int bit)   
{ 
   return m_algoDecisionInitial.at(bit); 
}
bool L1uGtAlgBlk::getAlgoDecisionPreScaled(int bit) 
{ 
   return m_algoDecisionPreScaled.at(bit); 
}
bool L1uGtAlgBlk::getAlgoDecisionFinal(int bit)     
{
   return m_algoDecisionFinal.at(bit); 
}


// reset the content of a L1uGtAlgBlk
void L1uGtAlgBlk::reset()
{

    //Clear out the header data
    m_orbitNr=0;
    m_bxNr=0;
    m_bxInEvent=0;
    m_finalOR=0;

    // Clear out the decision words
    // but leave the vector intact 
    m_algoDecisionInitial.assign(L1GlobalTriggerReadoutSetup::NumberPhysTriggers,false);
    m_algoDecisionPreScaled.assign(L1GlobalTriggerReadoutSetup::NumberPhysTriggers,false);
    m_algoDecisionFinal.assign(L1GlobalTriggerReadoutSetup::NumberPhysTriggers,false);


}

// pretty print the content of a L1uGtAlgBlk
void L1uGtAlgBlk::print(std::ostream& myCout) const
{

    
    myCout << " uGtAlgBlk: " << std::endl;
    
    myCout << "    Orbit Number (hex):  0x" << std::hex << std::setw(8) << std::setfill('0') << m_orbitNr << std::endl;

    myCout << "    Bx Number (hex):     0x" << std::hex << std::setw(4) << std::setfill('0') << m_bxNr << std::endl;

    myCout << "    Local Bx (hex):      0x" << std::hex << std::setw(1) << std::setfill('0') << m_bxInEvent << std::endl;

    myCout << "    Final OR (hex):      Ox" << std::hex << std::setw(1) << std::setfill('0') << m_finalOR << std::endl;
    
    // Loop through bits to create a hex word of algorithm bits.
    int lengthWd = m_algoDecisionInitial.size();
    myCout << "    Decision (Initial)   0x" << std::hex;
    int digit = 0;
    for(int i=lengthWd-1; i>-1; i--) {
      if(m_algoDecisionInitial.at(i)) digit |= (1 << (i%4));
      if((i%4) == 0){
         myCout << std::hex << std::setw(1) << digit;
	 digit = 0; 
      }  
    } //end loop over algorithm bits
    myCout << std::endl;
    
    // Loop through bits to create a hex word of algorithm bits.
    lengthWd = m_algoDecisionPreScaled.size();
    myCout << "    Decision (Prescaled) 0x" << std::hex;
    digit = 0;
    for(int i=lengthWd-1; i>-1; i--) {
      if(m_algoDecisionPreScaled.at(i)) digit |= (1 << (i%4));
      if((i%4) == 0){
         myCout << std::hex << std::setw(1) << digit;
	 digit = 0; 
      }  
    } //end loop over algorithm bits
    myCout << std::endl;


    // Loop through bits to create a hex word of algorithm bits.
    lengthWd = m_algoDecisionFinal.size();
    myCout << "    Decision (Final)     0x" << std::hex;
    digit = 0;
    for(int i=lengthWd-1; i>-1; i--) {
      if(m_algoDecisionFinal.at(i)) digit |= (1 << (i%4));
      if((i%4) == 0){
         myCout << std::hex << std::setw(1) << digit;
	 digit = 0; 
      }  
    } //end loop over algorithm bits
    myCout << std::endl;

}


