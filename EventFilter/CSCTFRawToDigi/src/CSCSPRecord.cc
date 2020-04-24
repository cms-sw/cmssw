#include <cstring>      // memcpy, bzero
#include "EventFilter/CSCTFRawToDigi/src/CSCSPRecord.h"
#include "EventFilter/CSCTFRawToDigi/src/CSCSPHeader.h"

bool CSCSPRecord::unpack(const unsigned short* &buf, unsigned int nonmasked_data_blocks, bool empty_blocks_suppressed,unsigned int tbin) throw() {
	memcpy((void*)this, buf, 8*sizeof(short));
	buf += 8;

	bool unpackError = check();

	bzero(me,15*sizeof(CSCSP_MEblock));
	bzero(mb, 2*sizeof(CSCSP_MBblock));
	bzero(me, 3*sizeof(CSCSP_SPblock));

	bzero(meFilled,sizeof(meFilled));
	bzero(mbFilled,sizeof(mbFilled));
	bzero(spFilled,sizeof(spFilled));

	const CSCSPHeader::ACTIVE id[] = {
		CSCSPHeader::F1, CSCSPHeader::F1, CSCSPHeader::F1,
		CSCSPHeader::F2, CSCSPHeader::F2, CSCSPHeader::F2,
		CSCSPHeader::F3, CSCSPHeader::F3, CSCSPHeader::F3,
		CSCSPHeader::F4, CSCSPHeader::F4, CSCSPHeader::F4,
		CSCSPHeader::F5, CSCSPHeader::F5, CSCSPHeader::F5
	};

	// 15 ME data blocks
	for(unsigned int block=0; block<15; block++)
		if( nonmasked_data_blocks & id[block] &&
			(!empty_blocks_suppressed || vp_bits&(1<<block)) ){
			unsigned int mpc = block/3, link = block%3;
			unpackError |= me[mpc][link].unpack(buf);
			me[mpc][link].tbin_         = tbin;
			me[mpc][link].valid_pattern = vp_bits&(1<<block);
			me[mpc][link].sync_error    = se_bits&(1<<block);
			me[mpc][link].sync_modified = sm_bits&(1<<block);
			me[mpc][link].alignment_fifo= af_bits&(1<<block);
			me[mpc][link].bxBit         = bx_bits&(1<<block);
			me[mpc][link].spInput_      = block+1;
			meFilled[mpc][link]         = true;
		}
	// 2 MB data blocks
	for(unsigned int block=0; block<2; block++)
		if( nonmasked_data_blocks & CSCSPHeader::DT &&
			(!empty_blocks_suppressed || (block?vq_b:vq_a) )){
			unpackError |= mb[block].unpack(buf);
			mb[block].tbin_         = tbin;
			mb[block].valid_quality = (block?vq_b:vq_a);
			mb[block].alignment_fifo= (block?af_barrel_2:af_barrel_1);
			mb[block].bxBit         = (block?bx_barrel_2:bx_barrel_1);
			mb[block].id_           =  block+1;
			mbFilled[block]         = true;
		}

	// 3 SP data blocks
	for(unsigned int block=0; block<3; block++)
		if( nonmasked_data_blocks & CSCSPHeader::SP &&
			(!empty_blocks_suppressed || (block==0?mode1:(block==1?mode2:mode3))) ){
			unpackError |= sp[block].unpack(buf);
			sp[block].tbin_ = tbin;
			sp[block].mode_ = (block==0?mode1:(block==1?mode2:mode3));
			sp[block].id_   = block+1;
			spFilled[block] = true;
		}

	return unpackError;
}

std::vector<CSCSP_MEblock> CSCSPRecord::LCTs(void) const throw() {
	std::vector<CSCSP_MEblock> result;
	for(int mpc=0; mpc<5; mpc++)
		for(int link=0; link<3; link++)
			if(meFilled[mpc][link]) result.push_back(me[mpc][link]);
	return result;
}

std::vector<CSCSP_MEblock> CSCSPRecord::LCTs(unsigned int mpc) const throw() {
	std::vector<CSCSP_MEblock> result;
	if( mpc<5 )
		for(int link=0; link<3; link++)
			if(meFilled[mpc][link]) result.push_back(me[mpc][link]);
	return result;
}

std::vector<CSCSP_MEblock> CSCSPRecord::LCT(unsigned int mpc, unsigned int link) const throw(){
	std::vector<CSCSP_MEblock> result;
	if( mpc<5 && link<3)
		if(meFilled[mpc][link]) result.push_back(me[mpc][link]);
	return result;
}

std::vector<CSCSP_SPblock> CSCSPRecord::tracks(void) const throw() {
	std::vector<CSCSP_SPblock> result;
	if(spFilled[0]) result.push_back(sp[0]);
	if(spFilled[1]) result.push_back(sp[1]);
	if(spFilled[2]) result.push_back(sp[2]);
	return result;
}

std::vector<CSCSP_MBblock> CSCSPRecord::mbStubs(void) const throw() {
	std::vector<CSCSP_MBblock> result;
	if(mbFilled[0]) result.push_back(mb[0]);
	if(mbFilled[1]) result.push_back(mb[1]);
	return result;
}

