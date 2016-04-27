/**
* \class GlobalAlgBlk
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
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"


// system include files


// user include files

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"


// constructors

// empty constructor, all members set to zero;
GlobalAlgBlk::GlobalAlgBlk(int orbitNr, int bxNr, int bxInEvent):
   m_orbitNr(orbitNr), m_bxNr(bxNr), m_bxInEvent(bxInEvent)
{

    //Clear out the header data
    m_finalOR=0;
    m_preScColumn=0;

    // Reserve/Clear out the decision words
    m_algoDecisionInitial.reserve(maxPhysicsTriggers);
    m_algoDecisionInitial.assign(maxPhysicsTriggers,false);
    
    m_algoDecisionPreScaled.reserve(maxPhysicsTriggers);
    m_algoDecisionPreScaled.assign(maxPhysicsTriggers,false);

    m_algoDecisionFinal.reserve(maxPhysicsTriggers);
    m_algoDecisionFinal.assign(maxPhysicsTriggers,false);

}


// empty constructor, all members set to zero;
GlobalAlgBlk::GlobalAlgBlk( )
{

    //Clear out the header data
    m_orbitNr=0;
    m_bxNr=0;
    m_bxInEvent=0;
    m_finalOR=0;
    m_finalORPreVeto = 0;
    m_finalORVeto = 0;    
    m_preScColumn=0;

    // Reserve/Clear out the decision words
    m_algoDecisionInitial.reserve(maxPhysicsTriggers);
    m_algoDecisionInitial.assign(maxPhysicsTriggers,false);
    
    m_algoDecisionPreScaled.reserve(maxPhysicsTriggers);
    m_algoDecisionPreScaled.assign(maxPhysicsTriggers,false);

    m_algoDecisionFinal.reserve(maxPhysicsTriggers);
    m_algoDecisionFinal.assign(maxPhysicsTriggers,false);

}



// destructor
GlobalAlgBlk::~GlobalAlgBlk()
{

    // empty now
}


/// Set decision bits
void GlobalAlgBlk::setAlgoDecisionInitial(unsigned int bit, bool val)   
{ 
   if(bit < m_algoDecisionInitial.size()) {
       
      m_algoDecisionInitial.at(bit) = val;   
   
   } else { 
     // Need some erorr checking here.
     LogTrace("L1TGlobal") << "Attempting to set an algorithm bit " << bit << " beyond limit " << m_algoDecisionInitial.size();
   }
   
}
void GlobalAlgBlk::setAlgoDecisionInterm(unsigned int bit, bool val) 
{ 

   if(bit < m_algoDecisionPreScaled.size()) {

     m_algoDecisionPreScaled.at(bit) = val; 
   } else { 
     // Need some erorr checking here.
     LogTrace("L1TGlobal") << "Attempting to set an algorithm bit " << bit << " beyond limit " << m_algoDecisionPreScaled.size();
   }

}
void GlobalAlgBlk::setAlgoDecisionFinal(unsigned int bit, bool val)     
{ 

   if(bit < m_algoDecisionFinal.size()) {
     m_algoDecisionFinal.at(bit) = val; 
   } else { 
     // Need some erorr checking here.
     LogTrace("L1TGlobal") << "Attempting to set an algorithm bit " << bit << " beyond limit " << m_algoDecisionFinal.size();
   }

}

/// Get decision bits
bool GlobalAlgBlk::getAlgoDecisionInitial(unsigned int bit) const  
{ 
   if(bit>=m_algoDecisionInitial.size()) return false;
   return m_algoDecisionInitial.at(bit); 
}
bool GlobalAlgBlk::getAlgoDecisionInterm(unsigned int bit) const
{ 
   if(bit>=m_algoDecisionPreScaled.size()) return false;
   return m_algoDecisionPreScaled.at(bit); 
}
bool GlobalAlgBlk::getAlgoDecisionFinal(unsigned int bit)  const   
{
   if(bit>=m_algoDecisionFinal.size()) return false;
   return m_algoDecisionFinal.at(bit); 
}


// reset the content of a GlobalAlgBlk
void GlobalAlgBlk::reset()
{

    //Clear out the header data
    m_orbitNr=0;
    m_bxNr=0;
    m_bxInEvent=0;
    m_finalOR=0;
    m_finalORPreVeto = 0;
    m_finalORVeto = 0;
    m_preScColumn=0;

    // Clear out the decision words
    // but leave the vector intact 
    m_algoDecisionInitial.assign(maxPhysicsTriggers,false);
    m_algoDecisionPreScaled.assign(maxPhysicsTriggers,false);
    m_algoDecisionFinal.assign(maxPhysicsTriggers,false);


}

// pretty print the content of a GlobalAlgBlk
void GlobalAlgBlk::print(std::ostream& myCout) const
{
    
    myCout << " uGtGlobalAlgBlk: " << std::endl;

    myCout << "    L1 Menu Name (hash):  0x" << std::hex << m_orbitNr << std::endl;

    myCout << "    L1 firmware (hash):   0x" << std::hex << m_bxNr << std::endl;

    myCout << "    Local Bx (hex):      0x" << std::hex << std::setw(1) << std::setfill('0') << m_bxInEvent << std::endl;
    
    myCout << "    PreScale Column:     "   <<std::setw(2) << m_preScColumn << std::endl;
    
    myCout << "    Final OR Veto:       " << std::hex << std::setw(1) << std::setfill('0') << m_finalORVeto << std::endl;
    
    myCout << "    Final OR:            " << std::hex << std::setw(1) << std::setfill('0') << m_finalOR << std::endl;
    
    // Loop through bits to create a hex word of algorithm bits.
    int lengthWd = m_algoDecisionInitial.size();
    myCout << "    Decision (Initial)   0x" << std::hex;
    int digit = 0;
    for(int i=lengthWd-1; i>-1; i--) {
      if(m_algoDecisionInitial.at(i)) digit |= (1 << (i%4));
      if((i%4) == 0){
         myCout << std::hex << std::setw(1) << digit;
	 digit = 0; 
	 if(i%32 == 0 && i<lengthWd-1) myCout << " ";
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
	 if(i%32 == 0 && i<lengthWd-1) myCout << " ";
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
	 if(i%32 == 0 && i<lengthWd-1) myCout << " ";
      }  
    } //end loop over algorithm bits
    myCout << std::endl;

}


