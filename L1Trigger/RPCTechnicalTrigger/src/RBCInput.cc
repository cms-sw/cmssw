// $Id: RBCInput.cc,v 1.3 2009/05/10 00:33:18 aosorio Exp $
// Include files 

// local
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCInput.h"

//-----------------------------------------------------------------------------
// Implementation file for class : RBCInput
//
// 2008-10-10 : Andres Osorio
//-----------------------------------------------------------------------------

std::istream& operator>>(std::istream &istr , RBCInput & rhs) {
  
  int _ks=0;
  
  for(int i=0 ; i < 30; ++i) { 
    istr >> rhs.input[i];
    if ( i < 15 ) _ks = 0;
    else _ks = 1;
    rhs.input_sec[_ks].set(i-(15*_ks), rhs.input[i]);
  }
  return istr;

}

std::ostream& operator<<(std::ostream &ostr , RBCInput & rhs) {
  
  for(int i=0; i < 15; ++i) ostr << rhs.input_sec[0][i];
  ostr << '\t';
  for(int i=0; i < 15; ++i) ostr << rhs.input_sec[1][i];
  ostr << '\n';
  
  return ostr;
  
}

void RBCInput::mask( const std::vector<int> & maskvec )
{
  
  //... operate on the first sector

  for(int i=0; i < 15; ++i) 
    if ( maskvec[i] ) input_sec[0].set(i,0);
  
  //... operate on the second sector
  
  for(int i=15; i < 30; ++i)
    if ( maskvec[i] ) input_sec[1].set( (i-15),0);
  
}

void RBCInput::force( const std::vector<int> & forcevec )
{
  
  if( m_debug ) std::cout << forcevec.size() << std::endl;

  std::bitset<15> tmp;
  
  for(int i=0; i < 15; ++i)
    tmp.set(i,forcevec[i]);
  
  //... operate on the first sector
  input_sec[0]|=tmp;
  tmp.reset();
  
  for(int i=15; i < 30; ++i)
    tmp.set( (i-15),forcevec[i]);
  
  input_sec[1]|=tmp;
  
  tmp.reset();
  
}

