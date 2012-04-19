#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <ostream>

const HcalDetId HcalDetId::Undefined(HcalEmpty,0,0,0);

HcalDetId::HcalDetId() : DetId() {
}

HcalDetId::HcalDetId(uint32_t rawid) : DetId(rawid) {
}

HcalDetId::HcalDetId(HcalSubdetector subdet, int tower_ieta, int tower_iphi, int depth) : DetId(Hcal,subdet) {
  // (no checking at this point!)
  id_ |= ((depth&0x1F)<<14) |
    ((tower_ieta>0)?(0x2000|(tower_ieta<<7)):((-tower_ieta)<<7)) |
    (tower_iphi&0x7F);
}

HcalDetId::HcalDetId(const DetId& gen) {
  if (!gen.null()) {
    HcalSubdetector subdet=(HcalSubdetector(gen.subdetId()));
    if (gen.det()!=Hcal || 
	(subdet!=HcalBarrel && subdet!=HcalEndcap && 
	 subdet!=HcalOuter && subdet!=HcalForward ))
      {
	throw cms::Exception("Invalid DetId") << "Cannot initialize HcalDetId from " << std::hex << gen.rawId() << std::dec; 
      }  
  }
  id_=gen.rawId();
}

HcalDetId& HcalDetId::operator=(const DetId& gen) {
  if (!gen.null()) {
    HcalSubdetector subdet=(HcalSubdetector(gen.subdetId()));
    if (gen.det()!=Hcal || 
	(subdet!=HcalBarrel && subdet!=HcalEndcap && 
	 subdet!=HcalOuter && subdet!=HcalForward )) {
      throw cms::Exception("Invalid DetId") << "Cannot assign HcalDetId from " << std::hex << gen.rawId() << std::dec; 
    }  
  }
  id_=gen.rawId();
  return (*this);
}

int HcalDetId::crystal_iphi_low() const { 
  int simple_iphi=((iphi()-1)*5)+1; 
  simple_iphi+=10;
  return ((simple_iphi>360)?(simple_iphi-360):(simple_iphi));
}

int HcalDetId::crystal_iphi_high() const { 
  int simple_iphi=((iphi()-1)*5)+5; 
  simple_iphi+=10;
  return ((simple_iphi>360)?(simple_iphi-360):(simple_iphi));
}

bool HcalDetId::validDetId( HcalSubdetector sd,
			    int             ies,
			    int             ip,
			    int             dp      ) {
  const int ie ( abs( ies ) ) ;

  return ( ( ip >=  1         ) &&
	   ( ip <= 72         ) &&
	   ( dp >=  1         ) &&
	   ( ie >=  1         ) &&
	   ( ( ( sd == HcalBarrel ) &&
	       ( ( ( ie <= 15         ) &&
		   ( dp <= maxDepthHB )    ) ||
		 ( ( ( ie == 16 ) ) && 
		   ( dp <= 2          )    ) ) ) ||
	     ( ( sd == HcalEndcap ) &&
	       ( ( ( ie >= 16 ) && ( ie <= 20 ) &&
		   ( dp <= maxDepthHE )    ) ||
		 ( ( ie >= 21 ) && ( ie <= 29 ) &&
		   ( dp <= maxDepthHE )    ) ) ) ||
	     (  ( sd == HcalOuter ) &&
		( ie <= 15 ) &&
		( dp ==  4 )           ) ||
	     (  ( sd == HcalForward ) &&
		( dp <=  2 )          &&
		( ( ( ie >= 29 ) &&  ( ie <= 39 ) &&
		    ( ip%2 == 1 )    ) ||
		  ( ( ie >= 40 ) &&  ( ie <= 41 ) &&
		    ( ip%4 == 3 )         )  ) ) ) );
}

bool HcalDetId::validDetIdPreLS1( HcalSubdetector sd,
				  int             ies,
				  int             ip,
				  int             dp      ) {
  const int ie ( abs( ies ) ) ;

  return ( ( ip >=  1         ) &&
	   ( ip <= 72         ) &&
	   ( dp >=  1         ) &&
	   ( ie >=  1         ) &&
	   ( ( ( sd == HcalBarrel ) &&
	       ( ( ( ie <= 14         ) &&
		   ( dp ==  1         )    ) ||
		 ( ( ( ie == 15 ) || ( ie == 16 ) ) && 
		   ( dp <= 2          )                ) ) ) ||
	     (  ( sd == HcalEndcap ) &&
		( ( ( ie == 16 ) &&
		    ( dp ==  3 )          ) ||
		  ( ( ie == 17 ) &&
		    ( dp ==  1 )          ) ||
		  ( ( ie >= 18 ) &&
		    ( ie <= 20 ) &&
		    ( dp <=  2 )          ) ||
		  ( ( ie >= 21 ) &&
		    ( ie <= 26 ) &&
		    ( dp <=  2 ) &&
		    ( ip%2 == 1 )         ) ||
		  ( ( ie >= 27 ) &&
		    ( ie <= 28 ) &&
		    ( dp <=  3 ) &&
		    ( ip%2 == 1 )         ) ||
		  ( ( ie == 29 ) &&
		    ( dp <=  2 ) &&
		    ( ip%2 == 1 )         )          )      ) ||
	     (  ( sd == HcalOuter ) &&
		( ie <= 15 ) &&
		( dp ==  4 )           ) ||
	     (  ( sd == HcalForward ) &&
		( dp <=  2 )          &&
		( ( ( ie >= 29 ) &&
		    ( ie <= 39 ) &&
		    ( ip%2 == 1 )    ) ||
		  ( ( ie >= 40 ) &&
		    ( ie <= 41 ) &&
		    ( ip%4 == 3 )         )  ) ) ) ) ;
}

int HcalDetId::hashed_index() const {

  const HcalSubdetector sd ( subdet()  ) ;
  const int             ip ( iphi()    ) ;
  const int             ie ( ietaAbs() ) ;
  const int             dp ( depth()   ) ;
  const int             zn ( zside() < 0 ? 1 : 0 ) ;

  int index(0);

  if (validDetIdPreLS1(sd,ie,ip,dp)) {
    // HB valid DetIds: phi=1-72,eta=1-14,depth=1; phi=1-72,eta=15-16,depth=1-2
    // HE valid DetIds: phi=1-72,eta=16-17,depth=1; phi=1-72,eta=18-20,depth=1-2; 
    //                  phi=1-71(in steps of 2),eta=21-26,depth=1-2; phi=1-71(in steps of 2),eta=27-28,depth=1-3
    //                  phi=1-71(in steps of 2),eta=29,depth=1-2
    // HO valid DetIds: phi=1-72,eta=1-15,depth=4!
    // HF valid DetIds: phi=1-71(in steps of 2),eta=29-39,depth=1-2; phi=3-71(in steps of 4),eta=40-41,depth=1-2

    index = ( ( sd == HcalBarrel ) ?
	      ( ip - 1 )*18 + dp - 1 + ie - ( ie<16 ? 1 : 0 ) + zn*kHBhalf :
	      ( ( sd == HcalEndcap ) ?
		2*kHBhalf + ( ip - 1 )*8 + ( ip/2 )*20 +
		( ( ie==16 || ie==17 ) ? ie - 16 :
		  ( ( ie>=18 && ie<=20 ) ? 2 + 2*( ie - 18 ) + dp - 1 :
		    ( ( ie>=21 && ie<=26 ) ? 8 + 2*( ie - 21 ) + dp - 1 :
		      ( ( ie>=27 && ie<=28 ) ? 20 + 3*( ie - 27 ) + dp - 1 :
			26 + 2*( ie - 29 ) + dp - 1 ) ) ) ) + zn*kHEhalf :
		( ( sd == HcalOuter ) ?
		  2*kHBhalf + 2*kHEhalf + ( ip - 1 )*15 + ( ie - 1 ) + zn*kHOhalf :
		  ( ( sd == HcalForward ) ?
		    2*kHBhalf + 2*kHEhalf + 2*kHOhalf + 
		    ( ( ip - 1 )/4 )*4 + ( ( ip - 1 )/2 )*22 + 
		    2*( ie - 29 ) + ( dp - 1 ) + zn*kHFhalf : -1 ) ) ) ) ; 
  } else if (validDetId(sd,ie,ip,dp)) {
    int depthHBPreLS1[15] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,2};
    int offHBDepth[15] = {0,(maxDepthHB-2), (2*maxDepthHB-4), (3*maxDepthHB-6),
			  (4*maxDepthHB-8),(5*maxDepthHB-10),(6*maxDepthHB-12),
			  (7*maxDepthHB-14),(8*maxDepthHB-16),(9*maxDepthHB-18),
			  (10*maxDepthHB-20),(11*maxDepthHB-22),(12*maxDepthHB-24),
			  (13*maxDepthHB-26),(14*maxDepthHB-28)};
    int depthHEPreLS1[14] = {3,1,2,2,2,2,2,2,2,2,2,3,3,2};
    int offHEDepth[14] = {0,(maxDepthHE-4), (2*maxDepthHE-6), (3*maxDepthHE-9),
			  (4*maxDepthHE-12),(5*maxDepthHE-15),(6*maxDepthHE-18),
			  (7*maxDepthHE-21),(8*maxDepthHE-24),(9*maxDepthHE-27),
			  (10*maxDepthHE-30),(11*maxDepthHE-33),(12*maxDepthHE-36),
			  (13*maxDepthHE-39)};
    if (sd == HcalBarrel) {
      index = kSizeForDenseIndexingPreLS1 + (ip-1)*(15*maxDepthHB-16) +
      offHBDepth[ie-1] + dp - depthHBPreLS1[ie-1] + ie-2 + zn*kHBHalfExtra;
    } else if (sd == HcalEndcap) {
      index = kSizeForDenseIndexingPreLS1 + kHBSizeExtra + 
      (ip-1)*(5*maxDepthHE-10)+ ( ip/2 )*(9*maxDepthHE-20) + 
      offHEDepth[ie-16] + dp - depthHEPreLS1[ie-16] + ie-17 + zn*kHEHalfExtra;
    }
  }

  return index;
}

HcalDetId HcalDetId::detIdFromDenseIndex( uint32_t di ) {

  if (validDenseIndex( di ) ) {
    HcalSubdetector sd ( HcalBarrel ) ;
    int ie ( 0 ) ;
    int ip ( 0 ) ;
    int dp ( 0 ) ;
    int in ( di ) ;
    int iz ( 1 ) ;
    if (validDenseIndexPreLS1( di )) {
      if ( in > 2*( kHBhalf + kHEhalf + kHOhalf ) - 1 ) { // HF
	sd  = HcalForward ;
	in -= 2*( kHBhalf + kHEhalf + kHOhalf ) ; 
	iz  = ( in<kHFhalf ? 1 : -1 ) ;
	in %= kHFhalf ; 
	ip  = 4*( in/48 ) ;
	in %= 48 ;
	ip += 1 + ( in>21 ? 2 : 0 ) ;
	if( 3 == ip%4 ) in -= 22 ;
	ie  = 29 + in/2 ;
	dp  = 1 + in%2 ;
      } else {
	if ( in > 2*( kHBhalf + kHEhalf ) - 1 ) { // HO
	  sd  = HcalOuter ;
	  in -= 2*( kHBhalf + kHEhalf ) ; 
	  iz  = ( in<kHOhalf ? 1 : -1 ) ;
	  in %= kHOhalf ; 
	  dp  = 4 ;
	  ip  = 1 + in/15 ;
	  ie  = 1 + ( in - 15*( ip - 1 ) ) ;
	} else {
	  if ( in > 2*kHBhalf - 1 ) { // Endcap
	    sd  = HcalEndcap ;
	    in -= 2*kHBhalf ;
	    iz  = ( in<kHEhalf ? 1 : -1 ) ;
	    in %= kHEhalf ; 
	    ip  = 2*( in/36 ) ;
	    in %= 36 ;
	    ip += 1 + in/28 ;
	    if( 0 == ip%2 ) in %= 28 ;
	    ie  = 15 + ( in<2 ? 1 + in : 2 + 
			 ( in<20 ? 1 + ( in - 2 )/2 : 9 +
			   ( in<26 ? 1 + ( in - 20 )/3 : 3 ) ) ) ;
	    dp  = ( in<1 ? 3 :
		    ( in<2 ? 1 : 
		      ( in<20 ? 1 + ( in - 2 )%2 : 
			( in<26 ? 1 + ( in - 20 )%3 : 
			  ( 1 + ( in - 26 )%2 ) ) ) ) ) ;
	  } else { // barrel
	    iz  = ( di<kHBhalf ? 1 : -1 ) ;
	    in %= kHBhalf ; 
	    ip = in/18 + 1 ;
	    in %= 18 ;
	    if ( in < 14 ) {
	      dp = 1 ;
	      ie = in + 1 ;
	    } else {
	      in %= 14 ;
	      dp =  1 + in%2 ;
	      ie = 15 + in/2 ;
	    }
	  }
	}
      }
    } else {
      int depthHBPreLS1[15] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,2};
      int offHBDepth[15] = {0,(maxDepthHB-2), (2*maxDepthHB-4), (3*maxDepthHB-6),
			    (4*maxDepthHB-8),(5*maxDepthHB-10),(6*maxDepthHB-12),
			    (7*maxDepthHB-14),(8*maxDepthHB-16),(9*maxDepthHB-18),
			    (10*maxDepthHB-20),(11*maxDepthHB-22),(12*maxDepthHB-24),
			    (13*maxDepthHB-26),(14*maxDepthHB-28)};
      int depthHEPreLS1[14] = {3,1,2,2,2,2,2,2,2,2,2,3,3,2};
      int offHEDepth[14] = {0,(maxDepthHE-4), (2*maxDepthHE-6), (3*maxDepthHE-9),
			    (4*maxDepthHE-12),(5*maxDepthHE-15),(6*maxDepthHE-18),
			    (7*maxDepthHE-21),(8*maxDepthHE-24),(9*maxDepthHE-27),
			    (10*maxDepthHE-30),(11*maxDepthHE-33),(12*maxDepthHE-36),
			    (13*maxDepthHE-39)};
      in -= kSizeForDenseIndexingPreLS1;
      if ( in > kHBSizeExtra - 1 ) { // Endcap
	sd  = HcalEndcap ;
	in -= kHBSizeExtra ;
	iz  = (in<kHEHalfExtra ? 1 : -1 ) ;
	in %= kHEHalfExtra; 
	ip  = 2*(in/(19*maxDepthHE-40) ) ;
	in %= (19*maxDepthHE-40);
	ip += (1 + in/(14*maxDepthHE-28));
	if ( 0 == ip%2 ) in %= (14*maxDepthHE-28);
	for (int i=13; i>=0; i--) {
	  int ioff = offHEDepth[i] + i - 1;
	  if (in > ioff) {
	    ie = i + 16;
	    dp = in - ioff + depthHEPreLS1[i];
	    break;
	  }
	}
      } else { // Barrel
	sd  = HcalBarrel;
	iz  = ( in<kHBHalfExtra ? 1 : -1 );
	in %= kHBHalfExtra; 
	ip  = ( in/(15*maxDepthHB-16) ) + 1;
	in %= (15*maxDepthHB-16);
	for (int i=14; i >= 0; --i) {
	  int ioff = offHBDepth[i] + i - 1;
	  if (in > ioff) {
	    ie = i + 1;
	    dp = in - ioff + depthHBPreLS1[i];
	    break;
	  }
	}
      }
    }
    return HcalDetId( sd, iz*int(ie), ip, dp ) ;
  } else {
    return HcalDetId() ;
  }
}

std::ostream& operator<<(std::ostream& s,const HcalDetId& id) {
  switch (id.subdet()) {
  case(HcalBarrel) : return s << "(HB " << id.ieta() << ',' << id.iphi() << ',' << id.depth() << ')';
  case(HcalEndcap) : return s << "(HE " << id.ieta() << ',' << id.iphi() << ',' << id.depth() << ')';
  case(HcalForward) : return s << "(HF " << id.ieta() << ',' << id.iphi() << ',' << id.depth() << ')';
  case(HcalOuter) : return s << "(HO " << id.ieta() << ',' << id.iphi() << ')';
  default : return s << id.rawId();
  }
}
