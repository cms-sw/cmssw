/** \file
 *
 *  $Date: 2005/10/24 15:56:19 $
 *  $Revision: 1.4 $
 */

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <FWCore/Utilities/interface/Exception.h>
#include <iostream>

using namespace std;

CSCDetId::CSCDetId():DetId(DetId::Muon, MuonSubdetId::CSC){}


CSCDetId::CSCDetId(uint32_t id):DetId(id) {
  if (det()!=DetId::Muon || subdetId()!=MuonSubdetId::CSC) {
    throw cms::Exception("InvalidDetId") << "CSCDetId ctor:"
					 << " det: " << det()
					 << " subdet: " << subdetId()
					 << " is not a valid CSC id";  
  }
}


CSCDetId::CSCDetId( int iendcap, int istation, int iring, int ichamber, 
		    int ilayer ) : 
  DetId(DetId::Muon, MuonSubdetId::CSC) 
{    
  if (iendcap  < minEndcapId  || iendcap  > maxEndcapId ||
      istation < minStationId || istation > maxStationId ||
      iring    < minRingId    || iring    > maxRingId ||
      ichamber < minChamberId || ichamber > maxChamberId ||
      ilayer   < minLayerId   || ilayer   > maxLayerId) {
    throw cms::Exception("InvalidDetId") << "CSCDetId ctor:" 
					 << " Invalid parameters: " 
					 << " E:"<< iendcap
					 << " S:"<< istation
					 << " R:"<< iring
					 << " C:"<< ichamber
					 << " L:"<< ilayer   
					 << std::endl;
  }
  id_ |= init(iendcap, istation, iring, ichamber, ilayer);
}

//bool CSCDetId::operator== (const CSCDetId& id) const
//{ 
//  return ( id_ == id.id_ );
//}

//bool CSCDetId::operator != (const CSCDetId& id) const
//{ 
//  return ( id_ != id.id_ );
//}

//bool CSCDetId::operator < (const CSCDetId& id) const
//{ 
//  if ( id_ < id.id_ ) 
//     return true;
//  else
//     return false;
//}

ostream& operator<<( ostream& os, const CSCDetId& id )
{
  // Note that there is no endl to end the output

   os << " E:" << id.endcap()
      << " S:" << id.station()
      << " R:" << id.ring()
      << " C:" << id.chamber()
      << " L:" << id.layer();
   return os;
}  

int CSCDetId::triggerSector() const
{

  int result;
  int ring    = this->ring();
  int station = this->station();
  int chamber = this->chamber();

  // This version 16-Nov-99 ptc to match simplified chamber labelling for cms116
  //@@ REQUIRES UPDATE TO 2005 REALITY, ONCE I UNDERSTAND WHAT THAT IS
  if(station > 1 && ring > 1 ) {
    result = (chamber+5) / 6; // ch 1-6->1, 7-12->2, ...
  }
  else {
    result = (chamber+2) / 3; // ch 1-3-> 1, 4-6->2, ...
  }
  return result;
}

int CSCDetId::triggerCscId() const 
{
  int result;
  int ring    = this->ring();
  int station = this->station();
  int chamber = this->chamber();

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

