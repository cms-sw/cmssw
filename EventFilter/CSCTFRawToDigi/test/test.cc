#include <string.h>
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <set>
#include "EventFilter/CSCTFRawToDigi/src/CSCTFEvent.cc"
#include "EventFilter/CSCTFRawToDigi/src/CSCSPEvent.cc"
#include "EventFilter/CSCTFRawToDigi/src/CSCSPRecord.cc"
#include "IORawData/CSCCommissioning/src/FileReaderDDU.cc"
//g++ -o test test.cc -I../../../

int main(int argc, char *argv[]){
	using namespace std;

	FILE *out;
	if( (out=fopen("dump2.raw","wt"))==NULL ){
		printf("Cannot open output file: %s (errno=%d)\n","dump2.raw",errno);
		return 1;
	}

	// DDU File Reader
	FileReaderDDU reader;
	reader.open(argv[1]);

	// Event buffer
	size_t size, nevents=0;
	const unsigned short *buf=0;

	// Main cycle
	while( (size = reader.read(buf)) /*&& nevents<100*/ ){
		unsigned short event[size];

		// Swep out C-words
		unsigned int index1=12, index2=12;
		memcpy(event,buf,12*sizeof(unsigned short));
		while( index1 < size ){
			if( (buf[index1]&0xF000)!=0xC000 ){
				event[index2] = buf[index1];
				index1++;
				index2++;
			} else {
				index1++;
			}
		}

		CSCTFEvent tfEvent, qwe;
		if(nevents%1000==0) cout<<"Event: "<<nevents<<endl;
///		cout<<" Unpack: "<<
		tfEvent.unpack(event,index2);
///		<<endl;

		bool log_event = false;

		vector<CSCSPEvent> SPs = tfEvent.SPs();
		for(int sp=0; sp<SPs.size(); sp++){
///			cout<<" L1A="<<SPs[0].header().L1A()<<endl;
			for(unsigned int tbin=0; tbin<SPs[sp].header().nTBINs(); tbin++){
				bool mismatch = false;
				vector<CSCSP_SPblock> tracks = SPs[sp].record(tbin).tracks();
				for(vector<CSCSP_SPblock>::const_iterator track=tracks.begin(); track!=tracks.end(); track++){
					unsigned int nStations=0;
					if( track->ME1_id() ) nStations++;
					if( track->ME2_id() ) nStations++;
					if( track->ME3_id() ) nStations++;
					if( track->ME4_id() ) nStations++;
					if( track->LCTs().size() != nStations ){
						mismatch = true;
						cout<<" mismatch found in tbin="<<tbin<<": ("<<track->LCTs().size()<<"!="<<nStations<<")";
					}

					if( mismatch ){
						cout<<hex<<" ME1: 0x"<<track->ME1_id()<<" tbin: "<<track->ME1_tbin()<<", "<<dec;
						cout<<hex<<" ME2: 0x"<<track->ME2_id()<<" tbin: "<<track->ME2_tbin()<<", "<<dec;
						cout<<hex<<" ME3: 0x"<<track->ME3_id()<<" tbin: "<<track->ME3_tbin()<<", "<<dec;
						cout<<hex<<" ME4: 0x"<<track->ME4_id()<<" tbin: "<<track->ME4_tbin()<<" "<<dec;
					}
				}
				vector<CSCSP_MEblock> lct = SPs[sp].record(tbin).LCTs();
				if( lct.size() ){
					cout<<"Event: "<<nevents<<" SP"<<sp<<" L1A="<<SPs[sp].header().L1A()<<" BXN="<<SPs[sp].header().BXN()<<" Orbit counter="<<SPs[sp].counters().orbit_counter()<<endl;
					cout<<" Endcap: "<<(SPs[sp].header().endcap()?2:1)<<" sector: "<<SPs[sp].header().sector();
					cout<<"  tbin: "<<tbin<<"  nLCTs: "<<SPs[sp].record(tbin).LCTs().size()<<" (";//<<endl;
				}
				for(std::vector<CSCSP_MEblock>::const_iterator i=lct.begin(); i!=lct.end(); i++){
					cout<<" F"<<((i->spInput()-1)/3+1)<<"/CSC"<<i->csc()<<":{w="<<i->wireGroup()<<",s="<<i->strip()<<"} ";
				}
				if( lct.size() ) cout<<" )"<<endl;
				std::vector<CSCSP_SPblock> trks = SPs[sp].record(tbin).tracks();
				if( trks.size() ){ cout<<"  Track(s) at BX=: "<<SPs[sp].header().BXN(); }
for(std::vector<CSCSP_SPblock>::const_iterator trk=trks.begin(); trk<trks.end(); trk++){ cout<<" mode="<<trk->mode(); if(trk->mode()==15 && SPs[sp].header().BXN()==380) log_event=true; }

				if( trks.size() ){ cout<<endl; }
			}
		}
		if(log_event) fwrite(event,2,index2,out); 
		nevents++;
	}
	fclose(out);

	return 0;
}
