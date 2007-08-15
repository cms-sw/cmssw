#include <DataFormats/MuonDetId/interface/CSCIndexer.h>
#include <iostream>
 
CSCIndexer::CSCIndexer(){
	// Fill the member vector which permits decoding of the linear index
	igorIndex.resize(217729);
	unsigned int id = 0;
	
	for(unsigned short int ie=1; ie!=3; ++ie){
	    for(unsigned short int is=1 ; is!=5; ++is){
		unsigned short int irmax = ringsInStation(is);
	      	for(unsigned short int ir=1; ir!=irmax+1; ++ir){
		    unsigned short int ischmax = stripChannelsPerLayer(is, ir);
		    unsigned short int icmax = chambersInRingOfStation(is, ir);
		    for(unsigned short int ic=1; ic!=icmax+1; ++ic){
			for(unsigned short int il=1; il!=7; ++il){
			    for(unsigned short int isch=1; isch!=ischmax+1; ++isch){
				id = ie*10000000 + is*1000000 + ir*100000 + ic*1000 + il*100 + isch;
			        // if ( isch==1 ){
				//   std::cout << "CSCIndexer ctor: id = " << id << std::endl;
				// }
				igorIndex[ stripChannelIndex(ie,is,ir,ic,il,isch) ] = id;
			    }
			}
		    }
		}
	    }
	}	

}

CSCDetId CSCIndexer::detIdFromLayer( unsigned int lin ) const {

	short linl = (lin-1)%6 + 1; // Turn layer 1-6 to 0-5 and then everything else is divisible by 6
	unsigned int chambersAll = (lin-linl)/6; // split off layer label part
	short line = chambersAll/234 + 1; // pick off endcap label
	unsigned int ix = chambersAll%234 + 1; // split off endcap label part

     // Station label 
	short lins = 0;
	if ( ix <= chambersUpToStation(1) ) lins = 1;
	else if ( ix <= chambersUpToStation(2) ) lins = 2;
	else if ( ix <= chambersUpToStation(3) )  lins = 3;
	else lins = 4;

     // Remove the station part
	unsigned iy = ix - chambersUpToStation(lins-1);
     
	short linr = 0;
	if ( iy <= chambersInStationUpToRing(lins, 1) )  linr = 1;
	else if ( iy <= chambersInStationUpToRing(lins, 2) ) linr = 2;
	else linr = 3;

     // Remove the ring part
	short linc = iy - chambersInStationUpToRing(lins, linr-1);

     //     std::cout << "Decoded id E" << line << " S" << lins << " R" << 
     //          linr << " C" << linc << " L" << linl << std::endl;

	return CSCDetId( line, lins, linr, linc, linl );
 }

CSCDetId CSCIndexer::detIdFromStrip( unsigned int sin ) const {
	unsigned int label = igorIndex.at( sin );
	unsigned short int isch = label%100;
	label -= isch;
	unsigned short int ie = label/10000000;
	label -= ie*10000000;
	unsigned short int is = label/1000000;
	label -= is*1000000;
	unsigned short int ir = label/100000;
	label -= ir*100000;
	unsigned short int ic = label/1000;
	label -= ic*1000;
	unsigned short int il = label/100;
	label -= il*100;
	//	if ( label != 0 ) {
	//		std::cout << "CSCIndexer:detIdFromStrip: failed to decode input=" << 
	//		sin << ", last value =" << label << std::endl;
	//	}
	return CSCDetId( ie, is, ir, ic, il );
}

unsigned short int CSCIndexer::stripChannel( unsigned int isch ) const { 
  //  std::cout << "stripChannel for isch=" << isch << " returns " << igorIndex.at(isch) << std::endl;
  return (igorIndex.at(isch)) % 100;
}

// The following implementation is superseded by more cryptic inline version...

/*
   unsigned short int CSCIndexer::chambersInStationUpToRing(short int is, short int ir) const {
     short nc = 0 ;
     if ( ir == 1 ) {
       if ( is == 1 )  nc = 36;
       else nc = 18;
     }
     else if ( ir == 2 ) {
       if ( is == 1 ) nc = 72;
       else nc = 54;
     }
     else if ( ir == 3 ) nc = 108;
     else nc = 0; // for ir==0

     return nc;   
  } 
*/
