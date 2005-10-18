/** \file
 *
 *  $Date: $
 *  $Revision: $
 */

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <iostream>

using namespace std;

bool CSCDetId::operator== (const CSCDetId& id) const
{ 
  return ( id_ == id.id_ );
}

bool CSCDetId::operator != (const CSCDetId& id) const
{ 
  return ( id_ != id.id_ );
}

bool CSCDetId::operator < (const CSCDetId& id) const
{ 
  if ( id_ < id.id_ ) 
     return true;
  else
     return false;
}

ostream& operator<<( ostream& os, const CSCDetId& id )
{
  // Note that there is no endl to end the output

   os << " E" << id.endcap()
      << " S" << id.station()
      << " R" << id.ring()
      << " C" << id.chamber()
      << " L" << id.layer();
   return os;
}  

unsigned int CSCDetId::sector() const
{

  unsigned int result;
  unsigned int ring    = this->ring();
  unsigned int station = this->station();
  unsigned int chamber = this->chamber();

  // This version 16-Nov-99 ptc to match simplified chamber labelling for cms116
  if(station > 1 && ring > 1 ) {
    result = (chamber+5) / 6; // ch 1-6->1, 7-12->2, ...
  }
  else {
    result = (chamber+2) / 3; // ch 1-3-> 1, 4-6->2, ...
  }
  return result;
}

unsigned int CSCDetId::cscId() const 
{
  unsigned int result;
  unsigned int ring    = this->ring();
  unsigned int station = this->station();
  unsigned int chamber = this->chamber();

  if( station == 1 ) {
    result = (chamber-1) % 3 + 1; // 1,2,3
    switch (ring) {
    case 1:
      break;
    case 2:
      result += 3; // 4,5,6
      break;
    case 3:
      result += 6; // 7,8,9
      break;
    }
  }
  else {
    if( ring == 1 ) {
      result = (chamber-1) % 3 + 1; // 1,2,3
    }
    else {
      result = (chamber-1) % 6 + 4; // 4,5,6,7,8,9
    }
  }
  return result;
}

