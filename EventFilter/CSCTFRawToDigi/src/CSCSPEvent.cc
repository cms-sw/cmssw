#include "EventFilter/CSCTFRawToDigi/src/CSCSPEvent.h"

bool CSCSPEvent::unpack(const unsigned short *&buf) throw() {
	bool unpackError = false;

	if( (buf[0]&0xF000) != 0x9000 || (buf[1]&0xF000) != 0x9000 || (buf[2]&0xF000) != 0x9000 || (buf[3]&0xF000) != 0x9000 ||
		(buf[4]&0xF000) != 0xA000 || (buf[5]&0xF000) != 0xA000 || (buf[6]&0xF000) != 0xA000 || (buf[7]&0xF000) != 0xA000 )
		return true;
	else
		unpackError |= header_.unpack(buf);

	if( !header_.empty() ){
		for(unsigned short tbin=0; tbin<header_.nTBINs(); tbin++)
			unpackError |= record_[tbin].unpack(buf,header_.active(),header_.suppression(),tbin);

		// Set LCTs for each track in each time bin
		for(unsigned short tbin=0; tbin<header_.nTBINs(); tbin++){
			for(unsigned short trk=0; trk<3; trk++){
				CSCSP_SPblock &track = record_[tbin].sp[trk];

				if( track.ME1_id() ){ // if track contains LCT from first station
					unsigned int mpc = ( track.ME1_id()>3 ? 1 : 0 );
					std::vector<CSCSP_MEblock> lcts = record_[track.ME1_tbin()].LCTs(mpc);
					for(std::vector<CSCSP_MEblock>::const_iterator lct=lcts.begin(); lct!=lcts.end(); lct++)
						if( lct->link()==(mpc?track.ME1_id()-3:track.ME1_id()) ){
							track.lct_[0] = *lct;
							track.lctFilled[0] = true;
						}
				}
				if( track.ME2_id() ){ // ... second station
					std::vector<CSCSP_MEblock> lcts = record_[track.ME2_tbin()].LCTs(2);
					for(std::vector<CSCSP_MEblock>::const_iterator lct=lcts.begin(); lct!=lcts.end(); lct++)
						if( lct->link()==track.ME2_id() ){
							track.lct_[1] = *lct;
							track.lctFilled[1] = true;
						}
				}
				if( track.ME3_id() ){ // ... third station
					std::vector<CSCSP_MEblock> lcts = record_[track.ME3_tbin()].LCTs(3);
					for(std::vector<CSCSP_MEblock>::const_iterator lct=lcts.begin(); lct!=lcts.end(); lct++)
						if( lct->link()==track.ME3_id() ){
							track.lct_[2] = *lct;
							track.lctFilled[2] = true;
						}
				}
				if( track.ME4_id() ){ // ... fourth station
					std::vector<CSCSP_MEblock> lcts = record_[track.ME4_tbin()].LCTs(4);
					for(std::vector<CSCSP_MEblock>::const_iterator lct=lcts.begin(); lct!=lcts.end(); lct++)
						if( lct->link()==track.ME4_id() ){
							track.lct_[3] = *lct;
							track.lctFilled[3] = true;
						}
				}
				if( track.MB_id() ){  // ... barrel
					std::vector<CSCSP_MBblock> stubs = record_[track.MB_tbin()+(track.MB_id()%2?0:1)].mbStubs();
					if( track.MB_id()>2 && stubs.size()!=2 ) ; // DQM error ?
					else {
						track.dt_ = stubs[(track.MB_id()-1)/2];
						track.dtFilled = true;
					}
				}

			}
		}
	}

	unpackError |= trailer_.unpack(buf);

	return unpackError;
}

