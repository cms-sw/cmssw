#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <cassert>

EcalTrigTowerDetId::EcalTrigTowerDetId() {
}
  
  
EcalTrigTowerDetId::EcalTrigTowerDetId(uint32_t rawid) : DetId(rawid) {
}

EcalTrigTowerDetId::EcalTrigTowerDetId(int zside, EcalSubdetector subDet, int i, int j, int mode)
  : DetId(Ecal,EcalTriggerTower) 
{
  int tower_i=0;
  int tower_j=0;

  if (mode == SUBDETIJMODE)
    {
      tower_i=i;
      tower_j=j;
    }
  else if (mode == SUBDETDCCTTMODE)
    {
      throw cms::Exception("InvalidDetId") << "EcalTriggerTowerDetId:  Cannot create object. SUBDETDCCTTMODE not yet implemented.";   
    }
  else
    throw cms::Exception("InvalidDetId") << "EcalTriggerTowerDetId:  Cannot create object.  Unknown mode for (int, EcalSubdetector, int, int) constructor.";
  
  if (tower_i > MAX_I || tower_i < MIN_I  || tower_j > MAX_J || tower_j < MIN_J)
    throw cms::Exception("InvalidDetId") << "EcalTriggerTowerDetId:  Cannot create object.  Indexes out of bounds.";
  
  id_|= ((zside>0)?(0x8000):(0x0)) | ((subDet == EcalBarrel)?(0x4000):(0x0)) | (tower_i<<7) | (tower_j & 0x7F);

}
  
EcalTrigTowerDetId::EcalTrigTowerDetId(const DetId& gen) 
{
  if (!gen.null() && ( gen.det()!=Ecal || gen.subdetId()!=EcalTriggerTower )) {
    throw cms::Exception("InvalidDetId");  }
  id_=gen.rawId();
}
  
EcalTrigTowerDetId& EcalTrigTowerDetId::operator=(const DetId& gen) {
  if (!gen.null() && ( gen.det()!=Ecal || gen.subdetId()!=EcalTriggerTower )) {
    throw cms::Exception("InvalidDetId");
  }
  id_=gen.rawId();
  return *this;
}

//New SM numbering scheme. Avoids discontinuity in phi crossing \eta=0  
int EcalTrigTowerDetId::iDCC() const 
{
  if ( subDet() == EcalBarrel )
    {
      //Correction since iphi is uniformized with HB convention 
      int iphi_simple = iphi() + 2 ;
      if (iphi_simple > 72 ) iphi_simple = iphi_simple % 72;
      int id = ( iphi_simple - 1 ) / kEBTowersInPhi + 1;
      if ( zside() < 0 ) id += 18;
      return id;
    }
  else
    throw cms::Exception("MethodNotImplemented") << "EcalTriggerTowerDetId: iDCC not yet implemented";
}

int EcalTrigTowerDetId::iTT() const 
{
  if ( subDet() == EcalBarrel )
    {
      int ie = ietaAbs() -1;
      int ip;
      int iphi_simple = iphi() + 2 ;
      if (iphi_simple > 72 )  iphi_simple = iphi_simple % 72;
      if (zside() < 0) {
	ip = (( iphi_simple -1 ) % kEBTowersInPhi ) + 1;
      } else {
	ip = kEBTowersInPhi - ((iphi_simple -1 ) % kEBTowersInPhi );
      }
      
      return (ie * kEBTowersInPhi) + ip;
    }
  else
    throw cms::Exception("MethodNotImplemented") << "EcalTriggerTowerDetId: iTT not yet implemented";
}

int EcalTrigTowerDetId::iquadrant() const
{
  if ( subDet() == EcalEndcap )
    return int((iphi()-1)/kEETowersInPhiPerQuadrant)+1;
  else
    throw cms::Exception("MethodNotApplicable") << "EcalTriggerTowerDetId: iquadrant not applicable";
}  

bool
EcalTrigTowerDetId::validDetId( int iz, EcalSubdetector sd , int i, int j )
{
   return ( 1 == abs( iz )               &&
	    0 <  i                       &&
	    0 <  j                       &&
	    kEETowersInPhiPerEndcap >= j &&
	    ( ( EcalBarrel     == sd &&
		kEBTowersInEta >=  i     ) ||
	      ( EcalEndcap     == sd &&
		kEEOuterEta    <=  i &&
		kEEInnerEta    >=  i &&
		( 27 > i ||
		  (  ( 0 >  iz  &&
		       0 == j%2    ) ||
		     ( 0 <  iz  &&
		       1 == j%2         ) ) ) ) ) ) ;
	    
}

int 
EcalTrigTowerDetId::hashedIndex() const 
{
   const unsigned int iea ( ietaAbs() ) ;
   const unsigned int iph ( iphi()    ) ;
   return ( subDet() == EcalBarrel  ? 
	    ( iDCC() - 1 )*kEBTowersPerSM + iTT() - 1 :
	    kEBTotalTowers + ( ( zside() + 1 )/2 )*kEETowersPerEndcap +
	    ( ( iea < 27 ? iea : 27 ) - kEEOuterEta )*kEETowersInPhiPerEndcap + 
	    ( iea < 27 ? iph : // for iphi=27,28 only half TT present, odd for EE-, even EE+
	      ( iea - 27 )*kEETowersInPhiPerEndcap/2 + ( iph + 1 )/2 ) - 1 ) ;
}

EcalTrigTowerDetId 
EcalTrigTowerDetId::detIdFromDenseIndex( uint32_t di ) 
{
   const EcalSubdetector sd ( di < kEBTotalTowers ? EcalBarrel : EcalEndcap ) ;
   const int iz ( di < kEBTotalTowers ? 
		  ( di < kEBHalfTowers ?  1 : -1 ) :
		  ( di - kEBTotalTowers < kEETowersPerEndcap ? -1 : 1 ) ) ;
   int i ;
   int j ;
   if( di < kEBTotalTowers ) // barrel
   {
      const unsigned int itt ( di%kEBTowersPerSM ) ;
      const unsigned int idc ( di/kEBTowersPerSM ) ;
      j = (idc%18)*kEBTowersInPhi + 
	 ( (1+iz)/2 )*kEBTowersInPhi - 
	 iz*(itt%kEBTowersInPhi)  + 1 - (1+iz)/2 - 2 ;
      if( j < 1 ) j += 72 ;
      i = 1 + itt/kEBTowersInPhi ;
   }
   else
   {
      const int eonly ( ( di - kEBTotalTowers )%kEETowersPerEndcap ) ;
      i = kEEOuterEta + eonly/kEETowersInPhiPerEndcap ;
      j = 1 + eonly%kEETowersInPhiPerEndcap ;
      if( 27 == i ) // last two rings have half of normal phi elementes
      {
	 if( j > kEETowersInPhiPerEndcap/2 )
	 {
	    ++i ; 
	    j -= kEETowersInPhiPerEndcap/2 ;
	 }
	 j = 2*j ;
	 if( 0 < iz ) --j ;
      }
   }
   assert( validDetId( iz, sd, i, j ) ) ;
   return EcalTrigTowerDetId( iz, sd, i, j ) ;
}

#include <ostream>
std::ostream& operator<<(std::ostream& s,const EcalTrigTowerDetId& id) {
  return s << "(EcalTT subDet " << ((id.subDet()==EcalBarrel)?("Barrel"):("Endcap")) 
	   <<  " iz " << ((id.zside()>0)?("+ "):("- ")) << " ieta " 
	   << id.ietaAbs() << " iphi " << id.iphi() << ')';
}

