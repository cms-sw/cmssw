#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
#include <iostream>

CSCRecHit2D::CSCRecHit2D() :
  theTpeak( -999. ),  
  thePositionWithinStrip(-999.),
  theErrorWithinStrip(-999.),
  theEnergyDeposit( -994. ),
  theQuality( 0 ), 
  theScaledWireTime( 0 ),
  theBadStrip( 0 ), 
  theBadWireGroup( 0 ),
  nStrips_(0),
  nWireGroups_(0),
  nTimeBins_(0),
  theLocalPosition(0.,0.), 
  theLocalError(0.,0.,0.)
{
  for ( unsigned int i=0; i< MAXSTRIPS; i++) theStrips_[i]=0;
  for ( unsigned int i=0; i< MAXSTRIPS; i++) 
    for ( unsigned int j=0; j< MAXTIMEBINS; j++) 
      theADCs_[i*MAXTIMEBINS+j]=0;
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
                          int scaledWireTime,
			  float energyDeposit):
  RecHit2DLocalPos( id ), 
  theTpeak( tpeak ),
  thePositionWithinStrip( posInStrip ),
  theErrorWithinStrip( errInStrip ),
  theEnergyDeposit( energyDeposit ),
  theQuality( quality ), 
  theScaledWireTime ( scaledWireTime ),
  theBadStrip( badStrip ), theBadWireGroup( badWireGroup ),
  theLocalPosition( pos ), 
  theLocalError( err )
{
  nStrips_=channels.size();
  nWireGroups_=wgroups.size();

  if ( nStrips_ > MAXSTRIPS ) {
    std::cout << "CSCRecHit2D: not enough strips in DataFormat! " << unsigned(nStrips_) <<  std::endl;
    nStrips_=MAXSTRIPS;
  }

  for ( unsigned int i=0; i< MAXSTRIPS; i++) theStrips_[i]=0;
  for ( unsigned int i=0; i< MAXSTRIPS; i++) theL1APhaseBits_[i]=0;
  for ( unsigned int i=0; i< MAXSTRIPS; i++) 
    for ( unsigned int j=0; j< MAXTIMEBINS; j++) 
      theADCs_[i*MAXTIMEBINS+j]=0;


  for ( unsigned int i=0; i<nStrips_; i++) {
    theStrips_[i]=channels[i] & 0x000000FF;
    theL1APhaseBits_[i]=channels[i] & 0x0000FF00;
  }
  if (nWireGroups_>0) {
    //take only the low bits
    hitWire_=wgroups[nWireGroups_/2] & 0x0000FFFF;
    theWGroupsBX_= (wgroups[nWireGroups_/2] >> 16)& 0x0000FFFF;
  }
  else {
    hitWire_=0;
    theWGroupsBX_=0;
  }
  ADCContainer tmp(adcs); //must be a bug in RangeMap!!!???
  nTimeBins_=tmp.size()/nStrips_;
  unsigned int k=0;
  for ( unsigned int i=0; i<nStrips_; i++)
    for ( unsigned int j=0; j<nTimeBins_; j++) {
      theADCs_[i*MAXTIMEBINS+j]=tmp[k];
      k++;
    }
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
  if (nStrips() == 0 && otherRecHit->nStrips() == 0 && nWireGroups() == 0 && otherRecHit->nWireGroups() == 0) return true;
  if ((what == allWires || what == someWires) && nWireGroups() == 0 && otherRecHit->nWireGroups() == 0) return true;
  if ((what == allStrips || what == someStrips) && nStrips() == 0 && otherRecHit->nStrips() == 0) return true;
  
  // Check to see if the wire containers are the same length
  if ((what == all || what == allWires) && nWireGroups() != otherRecHit->nWireGroups()) return false;
  
  // Check to see if the strip containers are the same length
  if ((what == all || what == allStrips) && nStrips() != otherRecHit->nStrips()) return false;
  
  bool foundWire = false;
  // Check to see if the wires are the same
  if (what != allStrips && what != someStrips) {
    //can we do better here?
    if ( hitWire() != otherRecHit->hitWire() )  return false;
  }
  
  // Check to see if the wires are the same
  bool foundStrip = false;
  if (what != allWires && what != someWires) {
    for (unsigned int i=0; i< nStrips(); i++) {
      bool found = false;
      for (unsigned int j=0; j< nStrips(); j++) {
	//a strip is a channel for all but ME1/1a chambers (where 3 ganged strips are a channel)
	if(cscDetId().channel(channels(i))==otherRecHit->cscDetId().channel(otherRecHit->channels(j))){
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


void CSCRecHit2D::print() const {
  std::cout << "CSCRecHit in CSC Detector: " << cscDetId() << std::endl;
  std::cout << "  local x = " << localPosition().x() << " +/- " << sqrt( localPositionError().xx() ) <<  " y = " << localPosition().y() << " +/- " << sqrt( localPositionError().yy() ) << std::endl;

  std::cout << " tpeak " << theTpeak << " psoInStrip " << thePositionWithinStrip << " errorinstrip " << theErrorWithinStrip << " " << " qual " << theQuality << " wiretime " << theScaledWireTime << " tbs " << theBadStrip << " bwg " << theBadWireGroup << std::endl; 

  std::cout << "  Channels: ";
  for (unsigned int i=0; i<nStrips(); i++) {std::cout << std::dec << channels(i) << " "
						      << " (" << "HEX: " << std::hex << channels(i) << ")" << " ";
  }
  std::cout << std::endl;
  
  
  /// L1A
  std::cout << "  L1APhase: ";
  for (int i=0; i<(int)nStrips(); i++) {
    std::cout << "|";
    for (int k=0; k<8 ; k++){ 
      std::cout << ((channelsl1a(i) >> (15-k)) & 0x1) << " ";}
    std::cout << "| ";       
  }           
  std::cout << std::endl;

  std::cout << "nWireGroups " << (int)nWireGroups() << " central wire " << hitWire_ <<std::endl;
}


std::ostream& operator<<(std::ostream& os, const CSCRecHit2D& rh) {
  os << "CSCRecHit2D: " <<
    "local x: " << rh.localPosition().x() << " +/- " << sqrt( rh.localPositionError().xx() ) <<
    " y: " << rh.localPosition().y() << " +/- " << sqrt( rh.localPositionError().yy() ) <<
    " in strip X: " << rh.positionWithinStrip() << " +/- " << rh.errorWithinStrip() <<
    " quality: " << rh.quality() << " tpeak: " << rh.tpeak() << " wireTime: " << rh.wireTime() << std::endl;
  os << "strips: ";
  for(size_t iS =0;iS< rh.nStrips();++iS){
    os <<rh.channels(iS)<<"  ";
  }
  int nwgs = rh.nWireGroups();
  if ( nwgs == 1 ) {
    os << "central wire: " << rh.hitWire() << " of " << nwgs << " wiregroup" << std::endl; }
  else { 
    os << "central wire: " << rh.hitWire() << " of " << nwgs << " wiregroups" << std::endl; }
  return os;
}


