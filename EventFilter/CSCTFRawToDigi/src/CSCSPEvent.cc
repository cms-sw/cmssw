#include "EventFilter/CSCTFRawToDigi/src/CSCSPEvent.h"
#include <map>
#include <list>

bool CSCSPEvent::unpack(const unsigned short *&buf) throw() {
	bool unpackError = false;

	if( (buf[0]&0xF000) != 0x9000 || (buf[1]&0xF000) != 0x9000 || (buf[2]&0xF000) != 0x9000 || (buf[3]&0xF000) != 0x9000 ||
		(buf[4]&0xF000) != 0xA000 || (buf[5]&0xF000) != 0xA000 || (buf[6]&0xF000) != 0xA000 || (buf[7]&0xF000) != 0xA000 )
		return true;
	else
		unpackError |= header_.unpack(buf);

	if( !header_.empty() ){
		// Block of Counters is added in format version 4.3 (dated by 05/27/2007)
		if( header_.format_version() )
			unpackError |= counters_.unpack(buf);

		if( header_.format_version()<3 || !header_.suppression() ){
			for(unsigned short tbin=0; tbin<header_.nTBINs(); tbin++)
				unpackError |= record_[tbin].unpack(buf,header_.active(),header_.suppression(),tbin);
		} else {
			for(unsigned short tbin=0, actual_tbin = (buf[7] >> 8) & 0x7; tbin<header_.nTBINs(); tbin++){
				bzero(&(record_[tbin]),sizeof(record_[tbin]));
				if( tbin != actual_tbin ) continue; // Skip supressed empty tbins in the format version >=5.3
				unpackError |= record_[tbin].unpack(buf,header_.active(),header_.suppression(),tbin);
				actual_tbin = (buf[7] >> 8) & 0x7;
			}
		}

		// Set LCTs for each track in each time bin
		for(unsigned short tbin=0; tbin<header_.nTBINs(); tbin++){
			for(unsigned short trk=0; trk<3; trk++){
				CSCSP_SPblock &track = record_[tbin].sp[trk];
				if( track.ME1_id()==0 && track.ME2_id()==0 && track.ME3_id()==0 && track.ME4_id()==0 && track.MB_id()==0 ) continue;
				// BXA algorithm is not trivial: first let's order all delays (MEx_tbin) aligning LCTs to one BX
				std::map< int, std::list<int> > timeline;
				if( track.ME1_id() ) timeline[track.ME1_tbin()].push_back(1);
				if( track.ME2_id() ) timeline[track.ME2_tbin()].push_back(2);
				if( track.ME3_id() ) timeline[track.ME3_tbin()].push_back(3);
				if( track.ME4_id() ) timeline[track.ME4_tbin()].push_back(4);
				int earliest_tbin=0, second_earliest_tbin=0;
				for(int bx=7; bx>=0; bx--){
					std::list<int>::const_iterator iter = timeline[bx].begin();
					while( iter != timeline[bx].end() ){
						if( earliest_tbin==0 ) earliest_tbin=bx;
						else if( second_earliest_tbin==0 ) second_earliest_tbin=bx;
						iter++;
					}
				}
				// MEx_tbin are LCTs delays shifting all of them to the bx of last LCT used to build a track
				//  let's convert delays to TBINs keeping in mind that tbin = some_shift - second_earliest_tbin
				if( track.ME1_id() ){ // if track contains LCT from first station
					unsigned int mpc = ( track.ME1_id()>3 ? 1 : 0 );
					int ME1_tbin = tbin - track.ME1_tbin() + second_earliest_tbin;
					if( ME1_tbin<0 || ME1_tbin>7 ) unpackError |= true;
					else {
						std::vector<CSCSP_MEblock> lcts = record_[ME1_tbin].LCTs(mpc);
						for(std::vector<CSCSP_MEblock>::const_iterator lct=lcts.begin(); lct!=lcts.end(); lct++)
							// Due to old MPC firmware link information was not accessible for some data:
							//if( lct->link()==(mpc?track.ME1_id()-3:track.ME1_id()) ){
							if( ((lct->spInput()-1)%3+1)==(mpc?track.ME1_id()-3:track.ME1_id()) ){
								track.lct_[0] = *lct;
								track.lctFilled[0] = true;
							}
					}
				}
				if( track.ME2_id() ){ // ... second station
					int ME2_tbin = tbin - track.ME2_tbin() + second_earliest_tbin;
					if( ME2_tbin<0 || ME2_tbin>7 ) unpackError |= true;
					else {
						std::vector<CSCSP_MEblock> lcts = record_[ME2_tbin].LCTs(2);
						for(std::vector<CSCSP_MEblock>::const_iterator lct=lcts.begin(); lct!=lcts.end(); lct++)
							// Due to old MPC firmware link information was not accessible for some data:
							//if( lct->link()==track.ME2_id() ){
							if( ((lct->spInput()-1)%3+1)==track.ME2_id() ){
								track.lct_[1] = *lct;
								track.lctFilled[1] = true;
							}
					}
				}
				if( track.ME3_id() ){ // ... third station
					int ME3_tbin = tbin - track.ME3_tbin() + second_earliest_tbin;
					if( ME3_tbin<0 || ME3_tbin>7 ) unpackError |= true;
					else {
						std::vector<CSCSP_MEblock> lcts = record_[ME3_tbin].LCTs(3);
						for(std::vector<CSCSP_MEblock>::const_iterator lct=lcts.begin(); lct!=lcts.end(); lct++)
							// Due to old MPC firmware link information was not accessible for some data:
							//if( lct->link()==track.ME3_id() ){
							if( ((lct->spInput()-1)%3+1)==track.ME3_id() ){
								track.lct_[2] = *lct;
								track.lctFilled[2] = true;
							}
					}
				}
				if( track.ME4_id() ){ // ... fourth station
					int ME4_tbin = tbin - track.ME4_tbin() + second_earliest_tbin;
					if( ME4_tbin<0 || ME4_tbin>7 ) unpackError |= true;
					else {
						std::vector<CSCSP_MEblock> lcts = record_[ME4_tbin].LCTs(4);
						for(std::vector<CSCSP_MEblock>::const_iterator lct=lcts.begin(); lct!=lcts.end(); lct++)
							// Due to old MPC firmware link information was not accessible for some data:
							//if( lct->link()==track.ME4_id() ){
							if( ((lct->spInput()-1)%3+1)==track.ME4_id() ){
								track.lct_[3] = *lct;
								track.lctFilled[3] = true;
							}
					}
				}
				if( track.MB_id() ){  // ... barrel
					if( (tbin==0 && track.MB_tbin()) || (tbin==6 && track.MB_id()%2==0) ) unpackError |= true;
					else {
						std::vector<CSCSP_MBblock> stubs = ( track.MB_id()%2==0 ? record_[tbin+1].mbStubs() : record_[tbin].mbStubs() );
						for(std::vector<CSCSP_MBblock>::const_iterator stub=stubs.begin(); stub!=stubs.end(); stub++)
							if( (stub->id()==1 && track.MB_id()<=2) || (stub->id()==2 && track.MB_id()>2) ){
								track.dt_ = *stub;
								track.dtFilled = true;
							}
						if( !track.dtFilled ) unpackError |= true;
					}
				}

			}
		}
	}

	unpackError |= trailer_.unpack(buf);

	return unpackError;
}

