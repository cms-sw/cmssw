#include "EventFilter/CSCTFRawToDigi/src/CSCTFEvent.h"
#include "EventFilter/CSCTFRawToDigi/src/CSCSPHeader.h"
#include <string.h>
#include <stdexcept>

unsigned int CSCTFEvent::unpack(const unsigned short *buf, unsigned int length) throw() {
	// Clean up
	nRecords = 0;
	bzero(sp,sizeof(sp));

	// Make sure that we are running right platform (can't imagine opposite)
	if( sizeof(unsigned long long)!=8 || sizeof(unsigned short)!=2 )
		throw std::runtime_error(std::string("Wrong platform: sizeof(unsigned long long)!=8 || sizeof(unsigned short)!=2"));

	// Type of coruptions
	unsigned long coruptions=0;

	// Combine 64-bit ddu word for simlicity and efficiency
	unsigned long long *dduWord = (unsigned long long*) buf, word_1=0, word_2=0;
	// 'length' counts ddu words now
	length /= sizeof(unsigned long long)/sizeof(unsigned short);
	// Set of markers
	bool spHeader=false, spTrailer=false;
	unsigned long spWordCount=0, spWordCountExpected=0;

	// Run through the buffer and check its structure
	unsigned int index=0;
	while( index<length ){
		word_1 = word_2;          // delay by 1 DDU word
		word_2 = dduWord[index];  // current DDU word

		if( spHeader && !spTrailer ) spWordCount++;

		if( (word_1&0xF000F000F000F000LL)==0x9000900090009000LL &&
			(word_2&0xF000F000F000F000LL)==0xA000A000A000A000LL ){
			if( spHeader ){
				coruptions |= MISSING_TRAILER;
				break;
			}
			spHeader=true;
			spTrailer=false;
			// number of 64-bit words between header and trailer
			spWordCount=0;
			spWordCountExpected=0;
			// need header to calculate expected word count
			CSCSPHeader header;
			const unsigned short *spWord = (unsigned short*) &dduWord[index-1];
			// we are here because we have a header => we are safe from crash instantiating one
			header.unpack(spWord);
			// calculate expected record length (internal variable 'shift' counts 16-bit words)
			for(unsigned short tbin=0,shift=0; tbin<header.nTBINs() && !header.empty(); tbin++){
				// check if didn't pass end of event, keep in mind that 'index' counts 64-bit words, and 'sp_record_length' - 16-bits
				if( length <= index+spWordCountExpected+1 ){
					coruptions |= OUT_OF_BUFFER;
					break;
				} else {
					// Data Block Header always exists
					spWordCountExpected += 2;
					// 15 ME data blocks
					for(unsigned int me_block=0; me_block<15; me_block++)
						if( header.active()&(1<<(me_block/3)) && (!header.suppression() || spWord[shift+0]&(1<<me_block)) )
							spWordCountExpected += 1;
					// 2 MB data blocks
					for(unsigned int mb_block=0; mb_block<2; mb_block++)
						if( header.active()&0x20 && (!header.suppression() || spWord[shift+1]&(1<<(mb_block+12))) )
							spWordCountExpected += 1;
					// 3 SP data blocks
					for(unsigned int sp_block=0; sp_block<3; sp_block++)
						if( header.active()&0x40 && (!header.suppression() || spWord[shift+1]&(0xF<<(sp_block*4))) )
							spWordCountExpected += 1;

					shift = spWordCountExpected*4;
				}
			}

			if( coruptions&OUT_OF_BUFFER ) break;
		}

		if( (word_1&0xF000F000F000F000LL)==0xF000F000F000F000LL &&
			(word_2&0xF000F000F000F000LL)==0xE000E000E000E000LL ){
			if( spTrailer ){
				coruptions |= MISSING_HEADER;
				break;
			}
			spHeader=false;
			spTrailer=true;

			if( spWordCount!=spWordCountExpected+2 ){
				coruptions |= WORD_COUNT;
				break;
			}
			// main unpacking
			const unsigned short *spWord = (unsigned short*) &dduWord[index-spWordCount-1];
			sp[nRecords++].unpack(spWord);
		}

		index++;
	}

	return coruptions;
}
