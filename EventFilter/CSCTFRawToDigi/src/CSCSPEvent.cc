#include "EventFilter/CSCTFRawToDigi/src/CSCSPEvent.h"
#include <map>
#include <list>
#include <iostream>

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
			// For the v.5.3 zero supression tbin has to be identified from the Data Block Header (BH2d word), as opposed to plain counting when empty records are not suppressed
			for(unsigned short tbin=0, actual_tbin = (buf[7] >> 8) & 0x7; tbin<header_.nTBINs(); tbin++){
				bzero(&(record_[tbin]),sizeof(record_[tbin]));
				// Check if we ran into the trailer (happens once all the records were read out)
				if( (buf[0]&0xF000)==0xF000 && (buf[1]&0xF000)==0xF000 && (buf[2]&0xF000)==0xF000 && (buf[3]&0xF000)==0xF000 &&
					(buf[4]&0xF000)==0xE000 && (buf[5]&0xF000)==0xE000 && (buf[6]&0xF000)==0xE000 && (buf[7]&0xF000)==0xE000 ) break;
				// Skip supressed empty tbins in the format version >=5.3
				if( tbin+1 != actual_tbin ) continue;
				// Unpack the record
				unpackError |= record_[tbin].unpack(buf,header_.active(),header_.suppression(),tbin);
				actual_tbin = (buf[7] >> 8) & 0x7;
			}
		}

		// Link initial LCTs to the tracks in each time bin
		for(unsigned short tbin=0; tbin<header_.nTBINs(); tbin++){
			for(unsigned short trk=0; trk<3; trk++){
				CSCSP_SPblock &track = record_[tbin].sp[trk];
				if( track.ME1_id()==0 && track.ME2_id()==0 && track.ME3_id()==0 && track.ME4_id()==0 && track.MB_id()==0 ) continue;
				// The key LCT identified by the BXA algorithm is the second earliest LCT
				int second_earliest_lct_delay = -1; // this is going to be a # tbins the key LCT was delayed to allign with the latest LCT
				if( track.mode() != 15 && track.mode() != 11 ){ // BXA works only on non halo tracks and non-singles
					// BXA algorithm is not trivial: first let's order all the delays (MEx_tbin), which are aligning LCTs to the tbin of the latest LCT
					std::map< int, std::list<int> > timeline;
					if( track.ME1_id() ) timeline[track.ME1_tbin()].push_back(1);
					if( track.ME2_id() ) timeline[track.ME2_tbin()].push_back(2);
					if( track.ME3_id() ) timeline[track.ME3_tbin()].push_back(3);
					if( track.ME4_id() ) timeline[track.ME4_tbin()].push_back(4);
					if( track.MB_id()  ) timeline[track.MB_tbin() ].push_back(5);
					int earliest_lct_delay = -1; //, second_earliest_lct_delay = -1;
					// Going from largest to smallest delay (earliest LCT pops up first in the loop)
					for(int delay=7; delay>=0 && second_earliest_lct_delay==-1; delay--){
						std::list<int>::const_iterator iter = timeline[delay].begin();
						while( iter != timeline[delay].end() && second_earliest_lct_delay==-1 ){
							if( earliest_lct_delay==-1 ) earliest_lct_delay=delay;
							else if( second_earliest_lct_delay==-1 ) second_earliest_lct_delay=delay;
							iter++;
						}
					}
				} else second_earliest_lct_delay = 0;

				// MEx_tbin are LCTs delays shifting all of them to the bx of last LCT used to build a track
				//  let's convert delays to TBINs keeping in mind that absolute_lct_tbin = track_tbin + (second_earliest_delay - lct_delay)
				if( track.ME1_id() ){ // if track contains LCT from the ME1
					unsigned int mpc = ( track.ME1_id()>3 ? 1 : 0 );
					int ME1_tbin = tbin + second_earliest_lct_delay - track.ME1_tbin();
///                                     if( track.ME1_tbin()>2 ) unpackError |= true; // because bxaDepth<=2
					if( ME1_tbin>=0 && ME1_tbin<7 ) {
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
				if( track.ME2_id() ){ // ... ME2
					int ME2_tbin = tbin + second_earliest_lct_delay - track.ME2_tbin();
///                                     if( track.ME2_tbin()>2 ) unpackError |= true; // because bxaDepth<=2
					if( ME2_tbin>=0 && ME2_tbin<7 ) {
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
				if( track.ME3_id() ){ // ... ME3
					int ME3_tbin = tbin + second_earliest_lct_delay - track.ME3_tbin();
///                                     if( track.ME3_tbin()>2 ) unpackError |= true; // because bxaDepth<=2
					if( ME3_tbin>=0 && ME3_tbin<7 ) {
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
					int ME4_tbin = tbin + second_earliest_lct_delay - track.ME4_tbin();
///                                     if( track.ME4_tbin()>2 ) unpackError |= true; // because bxaDepth<=2
					if( ME4_tbin>=0 && ME4_tbin<7 ) {
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
					int MB_tbin = tbin + second_earliest_lct_delay - track.MB_tbin();
///                                     if( track.MB_tbin()>2 ) unpackError |= true; // because bxaDepth<=2
					if( MB_tbin>=0 && MB_tbin<7 ) {
						std::vector<CSCSP_MBblock> stubs = record_[MB_tbin].mbStubs();
						for(std::vector<CSCSP_MBblock>::const_iterator stub=stubs.begin(); stub!=stubs.end(); stub++)
							if( (stub->id()==1 && track.MB_id()%2==1) || (stub->id()==2 && track.MB_id()%2==0) ){
								track.dt_ = *stub;
								track.dtFilled = true;
							}
					}
				}
			}
		}
	}
	unpackError |= trailer_.unpack(buf);

	return unpackError;
}

