#include "EventFilter/CSCTFRawToDigi/src/CSCSPEvent.h"

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

		for(unsigned short tbin=0; tbin<header_.nTBINs(); tbin++)
			unpackError |= record_[tbin].unpack(buf,header_.active(),header_.suppression(),tbin);

		// Set LCTs for each track in each time bin
		for(unsigned short tbin=0; tbin<header_.nTBINs(); tbin++){
			for(unsigned short trk=0; trk<3; trk++){
				CSCSP_SPblock &track = record_[tbin].sp[trk];

				if( track.ME1_id() ){ // if track contains LCT from first station
					unsigned int mpc = ( track.ME1_id()>3 ? 1 : 0 );
					if( tbin==0 && track.ME1_tbin() ) unpackError |= true;
					std::vector<CSCSP_MEblock> lcts = ( track.ME1_tbin() ? record_[tbin-1].LCTs(mpc) : record_[tbin].LCTs(mpc) );
					for(std::vector<CSCSP_MEblock>::const_iterator lct=lcts.begin(); lct!=lcts.end(); lct++)
						// Due to old MPC firmware link information was not accessible for some data:
						//if( lct->link()==(mpc?track.ME1_id()-3:track.ME1_id()) ){
						if( ((lct->spInput()-1)%3+1)==(mpc?track.ME1_id()-3:track.ME1_id()) ){
							track.lct_[0] = *lct;
							track.lctFilled[0] = true;
						}
				}
				if( track.ME2_id() ){ // ... second station
					if( tbin==0 && track.ME2_tbin() ) unpackError |= true;
					std::vector<CSCSP_MEblock> lcts = ( track.ME2_tbin() ? record_[tbin-1].LCTs(2) : record_[tbin].LCTs(2) );
					for(std::vector<CSCSP_MEblock>::const_iterator lct=lcts.begin(); lct!=lcts.end(); lct++)
						// Due to old MPC firmware link information was not accessible for some data:
						//if( lct->link()==track.ME2_id() ){
						if( ((lct->spInput()-1)%3+1)==track.ME2_id() ){
							track.lct_[1] = *lct;
							track.lctFilled[1] = true;
						}
				}
				if( track.ME3_id() ){ // ... third station
					if( tbin==0 && track.ME3_tbin() ) unpackError |= true;
					std::vector<CSCSP_MEblock> lcts = ( track.ME3_tbin() ? record_[tbin-1].LCTs(3) : record_[tbin].LCTs(3) );
					for(std::vector<CSCSP_MEblock>::const_iterator lct=lcts.begin(); lct!=lcts.end(); lct++)
						// Due to old MPC firmware link information was not accessible for some data:
						//if( lct->link()==track.ME3_id() ){
						if( ((lct->spInput()-1)%3+1)==track.ME3_id() ){
							track.lct_[2] = *lct;
							track.lctFilled[2] = true;
						}
				}
				if( track.ME4_id() ){ // ... fourth station
					if( tbin==0 && track.ME4_tbin() ) unpackError |= true;
					std::vector<CSCSP_MEblock> lcts = ( track.ME4_tbin() ? record_[tbin-1].LCTs(4) : record_[tbin].LCTs(4) );
					for(std::vector<CSCSP_MEblock>::const_iterator lct=lcts.begin(); lct!=lcts.end(); lct++)
						// Due to old MPC firmware link information was not accessible for some data:
						//if( lct->link()==track.ME4_id() ){
						if( ((lct->spInput()-1)%3+1)==track.ME4_id() ){
							track.lct_[3] = *lct;
							track.lctFilled[3] = true;
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

