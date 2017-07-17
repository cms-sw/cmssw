/**
* \class GlobalExtBlk
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
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"


// system include files


// user include files

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"


// constructors


// empty constructor, all members set to zero;
GlobalExtBlk::GlobalExtBlk( )
{

    // Reserve/Clear out the decision words
    m_extDecision.reserve(maxExternalConditions);
    m_extDecision.assign(maxExternalConditions,false);

}



// destructor
GlobalExtBlk::~GlobalExtBlk()
{

    // empty now
}


/// Set decision bits
void GlobalExtBlk::setExternalDecision(unsigned int bit, bool val)   
{ 
   if(bit < m_extDecision.size()) {
       
      m_extDecision.at(bit) = val;   
   
   }else { 
     // Need some erorr checking here.
     LogTrace("L1TGlobal") << "Attempting to set a external bit " << bit << " beyond limit " << m_extDecision.size();
   }
      
}


/// Get decision bits
bool GlobalExtBlk::getExternalDecision(unsigned int bit) const  
{ 
   if(bit>=m_extDecision.size()) return false;
   return m_extDecision.at(bit); 
}


// reset the content of a GlobalExtBlk
void GlobalExtBlk::reset()
{

    // Clear out the decision words
    // but leave the vector intact 
    m_extDecision.assign(maxExternalConditions,false);


}

// pretty print the content of a GlobalExtBlk
void GlobalExtBlk::print(std::ostream& myCout) const
{

    myCout << " GlobalExtBlk " << std::endl;

    // Loop through bits to create a hex word of algorithm bits.
    int lengthWd = m_extDecision.size();
    myCout << "    External Conditions   0x" << std::hex;
    int digit = 0;
    bool firstNonZero = false;
    for(int i=lengthWd-1; i>-1; i--) {
      if(m_extDecision.at(i)) digit |= (1 << (i%4));
      if(digit > 0) firstNonZero = true;
      if((i%4) == 0 && firstNonZero){
         myCout << std::hex << std::setw(1) << digit;
	 digit = 0; 
	 if(i%32 == 0 && i<lengthWd-1) myCout << " ";
      }  
    } //end loop over algorithm bits
    if(!firstNonZero) myCout << "0";
    myCout << std::endl;    

}


