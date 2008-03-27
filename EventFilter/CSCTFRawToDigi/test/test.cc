#include <string.h>
#include <iostream>
#include <iomanip>
#include <set>
#include "EventFilter/CSCTFRawToDigi/src/CSCTFEvent.cc"
#include "EventFilter/CSCTFRawToDigi/src/CSCSPEvent.cc"
#include "EventFilter/CSCTFRawToDigi/src/CSCSPRecord.cc"
#include "IORawData/CSCCommissioning/src/FileReaderDDU.cc"
//g++ -o test test.cc -I../../../

int main(int argc, char *argv[]){
	using namespace std;

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

		vector<CSCSPEvent> SPs = tfEvent.SPs();
		for(int sp=0; sp<SPs.size(); sp++){
///			cout<<" L1A="<<SPs[0].header().L1A()<<endl;
			for(unsigned int tbin=0; tbin<SPs[sp].header().nTBINs(); tbin++){
				vector<CSCSP_MEblock> lct = SPs[sp].record(tbin).LCTs();
				if( lct.size() ){
					cout<<"Event: "<<nevents<<" SP"<<sp<<" L1A="<<SPs[sp].header().L1A()<<endl;
					cout<<" Endcap: "<<(SPs[sp].header().endcap()?2:1)<<" sector: "<<SPs[sp].header().sector();
					cout<<"  tbin: "<<tbin<<"  nLCTs: "<<SPs[sp].record(tbin).LCTs().size()<<" (";//<<endl;
				}
				for(std::vector<CSCSP_MEblock>::const_iterator i=lct.begin(); i!=lct.end(); i++){
					cout<<" F"<<((i->spInput()-1)/3+1)<<"/CSC"<<i->csc()<<":{w="<<i->wireGroup()<<",s="<<i->strip()<<"} ";
				}
				if( lct.size() ) cout<<" )"<<endl;
			}
		}
		nevents++;
	}

	return 0;
}
