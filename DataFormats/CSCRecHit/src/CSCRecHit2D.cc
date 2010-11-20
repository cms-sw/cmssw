#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
#include <iostream>

CSCRecHit2D::CSCRecHit2D() :
  theLocalPosition(0.,0.), 
  theLocalError(0.,0.,0.),
  theStrips( ChannelContainer() ),
  theADCs( ADCContainer() ),
  theWireGroups( ChannelContainer() ),
  theTpeak( -999. ),  
  thePositionWithinStrip(-999.),
  theErrorWithinStrip(-999.),
  theQuality( 0 ), theBadStrip( 0 ), theBadWireGroup( 0 ),
  theScaledWireTime( 0 )
{

/// Extract low byte to account strip number
theStripsLowBits.clear();
for(int i=0; i<(int)theStrips.size(); i++){
        theStripsLowBits.push_back(theStrips[i] & 0x000000FF);
	}
/// Extract middle byte to account L1A phase
theStripsHighBits.clear();
for(int i=0; i<(int)theStrips.size(); i++){
        theStripsHighBits.push_back(theStrips[i] & 0x0000FF00);
	}

/// To account BX in wgroups
theWgroupsHighBits.clear();
for(int i=0; i<(int)theWireGroups.size(); i++)
   theWgroupsHighBits.push_back((theWireGroups[i] >> 16) & 0x0000FFFF);
theWgroupsLowBits.clear();
for(int i=0; i<(int)theWireGroups.size(); i++)
   theWgroupsLowBits.push_back(theWireGroups[i] & 0x0000FFFF);

}

CSCRecHit2D::CSCRecHit2D( const CSCDetId& id, 
                          const LocalPoint& pos, 
                          const LocalError& err, 
	                  const ChannelContainer& channels, 
                          const ADCContainer& adcs,
                          const ChannelContainer& wgroups,
                          float tpeak, 
			  float posInStrip, 
                          float errInStrip,
			  int quality, short int badStrip, short int badWireGroup,
                          int scaledWireTime ):
  RecHit2DLocalPos( id ), 
  theLocalPosition( pos ), 
  theLocalError( err ),
  theStrips( channels ),
  theADCs( adcs ),
  theWireGroups( wgroups ),
  theTpeak( tpeak ),
  thePositionWithinStrip( posInStrip ),
  theErrorWithinStrip( errInStrip ),
  theQuality( quality ), theBadStrip( badStrip ), theBadWireGroup( badWireGroup ),
  theScaledWireTime ( scaledWireTime )
{

/// Extract second byte to account strip number
theStripsLowBits.clear();
for(int i=0; i<(int)theStrips.size(); i++){
        theStripsLowBits.push_back(theStrips[i] & 0x000000FF);
	}
/// Extract middle byte to account L1A phase
theStripsHighBits.clear();
for(int i=0; i<(int)theStrips.size(); i++){
        theStripsHighBits.push_back(theStrips[i] & 0x0000FF00);
	}

/// To account BX in wgroups
theWgroupsHighBits.clear();
for(int i=0; i<(int)theWireGroups.size(); i++)
   theWgroupsHighBits.push_back((theWireGroups[i] >> 16) & 0x0000FFFF);
theWgroupsLowBits.clear();
for(int i=0; i<(int)theWireGroups.size(); i++)
   theWgroupsLowBits.push_back(theWireGroups[i] & 0x0000FFFF);

}

CSCRecHit2D::~CSCRecHit2D() {}

bool CSCRecHit2D::sharesInput(const TrackingRecHit *other, TrackingRecHit::SharedInputType what) const {
  
  // This is to satisfy the base class virtual function
  
  // @@ Cast the enum (!) But what if the TRH::SIT changes?!
  CSCRecHit2D::SharedInputType cscWhat = static_cast<CSCRecHit2D::SharedInputType>(what);
  return sharesInput(other, cscWhat);
}

bool CSCRecHit2D::sharesInput(const TrackingRecHit *other, CSCRecHit2D::SharedInputType what) const {
  
  // Check to see if the TrackingRecHit is actually a CSCRecHit2D.
  if (other->geographicalId().subdetId() != MuonSubdetId::CSC) return false;
  
  // Now I can static cast, because the previous guarantees that this is a CSCRecHit2D
  const CSCRecHit2D *otherRecHit = static_cast<const CSCRecHit2D *>(other);
  
  return sharesInput(otherRecHit, what);
}

bool CSCRecHit2D::sharesInput(const  CSCRecHit2D *otherRecHit, CSCRecHit2D::SharedInputType what) const {
  
  // Check to see if the geographical ID of the two are the same
  if (geographicalId() != otherRecHit->geographicalId()) return false;
  
  // Trivial cases
  const ChannelContainer otherStrips = otherRecHit->channels();
  const ChannelContainer otherWireGroups = otherRecHit->wgroups();
  if (theStrips.size() == 0 && otherStrips.size() == 0 && theWireGroups.size() == 0 && otherWireGroups.size() == 0) return true;
  if ((what == allWires || what == someWires) && theWireGroups.size() == 0 && otherWireGroups.size() == 0) return true;
  if ((what == allStrips || what == someStrips) && theStrips.size() == 0 && otherStrips.size() == 0) return true;
  
  // Check to see if the wire containers are the same length
  if ((what == all || what == allWires) && theWireGroups.size() != otherWireGroups.size()) return false;
  
  // Check to see if the strip containers are the same length
  if ((what == all || what == allStrips) && theStrips.size() != otherStrips.size()) return false;
  
  bool foundWire = false;
  // Check to see if the wires are the same
  if (what != allStrips && what != someStrips) {
    for (ChannelContainer::const_iterator iWire = theWireGroups.begin(); iWire != theWireGroups.end(); ++iWire) {
      bool found = false;
      for (ChannelContainer::const_iterator jWire = otherWireGroups.begin(); jWire != otherWireGroups.end(); ++jWire) {
	if (*iWire == *jWire) {
	  if (what == some || what == someWires) return true;
	  else {
	    found = true;
	    foundWire = true;
	    break;
	  }
	}
      }
      if ((what == all || what == allWires) && !found) return false;
    }
    if (what == someWires && !foundWire) return false;
  }
  
  // Check to see if the wires are the same
  bool foundStrip = false;
  if (what != allWires && what != someWires) {
    for (ChannelContainer::const_iterator iStrip = theStrips.begin(); iStrip != theStrips.end(); ++iStrip) {
      bool found = false;
      for (ChannelContainer::const_iterator jStrip = otherStrips.begin(); jStrip != otherStrips.end(); ++jStrip) {
	//a strip is a channel for all but ME1/1a chambers (where 3 ganged strips are a channel)
	if(cscDetId().channel(*iStrip)==otherRecHit->cscDetId().channel(*jStrip)){
	  if (what == some || what == someStrips) return true;
	  else {
	    found = true;
	    foundStrip = true;
	    break;
	  }
	}
      }
      if ((what == all || what == allStrips) && !found) return false;
    }
    if (what == someStrips && !foundStrip) return false;
  }
  
  // In case we were looking for "some" and found absolutely nothing.
  if (!foundWire && !foundStrip) return false;
  
  // If we made it this far, then:
  //  1) the detector IDs are the same
  //  2) the channel containers have the same number of entries
  //  3) for each entry in my channel container, I can find the same value in the other RecHit's corresponding channel container
  // I think that means we are the same.
  return true;
}

/// Prefered to have this version of print() for debugging;
/// will be removed soon
/*
void CSCRecHit2D::print() const {
std::cout << "CSCRecHit in CSC Detector: " << cscDetId() << std::endl;
std::cout << "  local x = " << localPosition().x() << " +/- " << sqrt( localPositionError().xx() ) <<  " y = " << localPosition().y() << " +/- " << sqrt( localPositionError().yy() ) << std::endl;

/// L1A
std::cout << "  L1A+Channels: ";
for (int i=0; i<(int)channelsTotal().size(); i++) {std::cout 
       << std::hex << channelsTotal()[i] << " ";}
std::cout << std::endl;

std::cout << "  Channels: ";
for (int i=0; i<(int)channels().size(); i++) {std::cout << std::dec << channels()[i] << " "
     << " (" << "HEX: " << std::hex << channels()[i] << ")" << " ";
      }
  std::cout << std::endl;


/// L1A
std::cout << "  L1APhase: ";
  for (int i=0; i<(int)channelsl1a().size(); i++) {
       std::cout << "|";
       for (int k=0; k<8 ; k++){ 
       std::cout << ((channelsl1a()[i] >> (15-k)) & 0x1) << " ";}
       std::cout << "| ";       
       }           
  std::cout << std::endl;

 
std::cout << "  WireGroups: ";
for (int i=0; i<(int)wgroups().size(); i++) {std::cout << std::dec << wgroups()[i]  
       << " ";}
  std::cout << std::endl;
}
*/


void CSCRecHit2D::print() const {
std::cout << "CSCRecHit in CSC Detector: " << cscDetId() << std::endl;
std::cout << "  local x = " << localPosition().x() << " +/- " << sqrt( localPositionError().xx() ) <<  " y = " << localPosition().y() << " +/- " << sqrt( localPositionError().yy() ) << std::endl;

/// L1A
std::cout << "  L1A+Channels: ";
for (int i=0; i<(int)channelsTotal().size(); i++) {std::cout 
       << std::hex << channelsTotal()[i] << " ";}
std::cout << std::endl;

std::cout << "  Channels: ";
for (int i=0; i<(int)channels().size(); i++) {std::cout << std::dec << channels()[i] << " "
     << " (" << "HEX: " << std::hex << channels()[i] << ")" << " ";
      }
  std::cout << std::endl;


/// L1A
std::cout << "  L1APhase: ";
  for (int i=0; i<(int)channelsl1a().size(); i++) {
       std::cout << "|";
       for (int k=0; k<8 ; k++){ 
       std::cout << ((channelsl1a()[i] >> (15-k)) & 0x1) << " ";}
       std::cout << "| ";       
       }           
  std::cout << std::endl;

std::cout << "  BX + WireGroups combined: ";
   for (int i=0; i<(int)wgroupsBXandWire().size(); i++) {std::cout //std::dec << wgroups()[i] 
        << "HEX: " << std::hex << wgroupsBXandWire()[i] << std::hex << " | ";
   }
 
std::cout << "  WireGroups: ";
for (int i=0; i<(int)wgroups().size(); i++) {std::cout << std::dec << wgroups()[i]  
       << " | ";}
std::cout << " BX#: ";
   for (int i=0; i<(int)wgroupsBX().size(); i++) {std::cout << std::dec << wgroupsBX()[i] 
       << " (" << "HEX: " << std::hex << wgroupsBX()[i] << ")" << " | ";
   }

  std::cout << std::endl;
}


/*
void CSCRecHit2D::print() const {
  std::cout << *this << std::endl;
}
*/

std::ostream& operator<<(std::ostream& os, const CSCRecHit2D& rh) {
  os << "CSCRecHit2D: local x = " << rh.localPosition().x() << " +/- " << sqrt( rh.localPositionError().xx() ) <<
    " y = " << rh.localPosition().y() << " +/- " << sqrt( rh.localPositionError().yy() ) <<
    " in strip X = " << rh.positionWithinStrip() << " +/-  = " << rh.errorWithinStrip()<<" quality = "<<rh.quality()<<"\n";
  os << "             strip # : ";
  for(size_t iS =0;iS< rh.channels().size();++iS){
    os <<rh.channels().at(iS)<<"  ";
  }
os << "\n             wire # : ";
  for(size_t iW =0;iW< rh.wgroups().size();++iW){
    os <<rh.wgroups().at(iW)<<"  ";
  }
  return os;
}


