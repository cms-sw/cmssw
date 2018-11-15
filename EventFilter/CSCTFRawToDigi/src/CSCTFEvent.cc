#include "EventFilter/CSCTFRawToDigi/src/CSCTFEvent.h"
#include "EventFilter/CSCTFRawToDigi/src/CSCSPHeader.h"
#include <cstring>
#include <stdexcept>

unsigned int CSCTFEvent::unpack(const unsigned short *buf, unsigned int length) {
	// Clean up
	nRecords = 0;
	bzero(sp,sizeof(sp));

	// Make sure that we are running right platform (can't imagine opposite)
	if( sizeof(unsigned long long)!=8 || sizeof(unsigned short)!=2 )
		throw std::runtime_error(std::string("Wrong platform: sizeof(unsigned long long)!=8 || sizeof(unsigned short)!=2"));

	// Type of coruptions
	unsigned long coruptions=0;

	// Combine 64-bit ddu word for simlicity and efficiency
	const unsigned long long *dduWord = reinterpret_cast<const unsigned long long*>(buf);
	unsigned long long word_1=0, word_2=0;
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
			const unsigned short *spWord = reinterpret_cast<const unsigned short*>(&dduWord[index-1]);
			// we are here because we have a header => we are safe from crash instantiating one
			header.unpack(spWord);

			// Counter block exists only in format version 4.3 and higher
			if( header.format_version() && !header.empty() ){
				if( length > index+1 ){ spWord += 4; } else { coruptions |= OUT_OF_BUFFER; break; }
			}

			// Calculate expected record length (internal variable 'shift' counts 16-bit words)
			for(unsigned short tbin=0,shift=0; tbin<header.nTBINs() && !header.empty(); tbin++){
				// check if didn't pass end of event, keep in mind that 'length', 'index', and 'spWordCountExpected' counts 64-bit words
				if( length <= index+spWordCountExpected+2 ){
					coruptions |= OUT_OF_BUFFER;
					break;
				} else {
					// In the format version >=5.3 with zero_supression
					if( header.format_version()>=3 && header.suppression() ){
						//  we seek the loop index untill it matches the current non-empty tbin
						if( ((spWord[shift+7]>>8) & 0x7) != tbin+1 ) continue;
						//  otherwise there may be no more non-empty tbins and we ran into the trailer
						if( (spWord[shift+0]&0xF000)==0xF000 && (spWord[shift+1]&0xF000)==0xF000 && (spWord[shift+2]&0xF000)==0xF000 && (spWord[shift+3]&0xF000)==0xF000 &&
							(spWord[shift+4]&0xF000)==0xE000 && (spWord[shift+5]&0xF000)==0xE000 && (spWord[shift+6]&0xF000)==0xE000 && (spWord[shift+7]&0xF000)==0xE000 ) break;
					}

					// Data Block Header always exists if we got so far
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

					shift = spWordCountExpected*4; // multiply by 4 because 'shift' is a 16-bit array index and 'spWordCountExpected' conuts 64-bit words
				}
			}

			// Counter block exists only in format version 4.3 and higher
			if( header.format_version() && !header.empty() ) spWordCountExpected += 1;

			if( coruptions&OUT_OF_BUFFER ) break;
		}

		//if( !spHeader && spTrailer ) coruptions |= NONSENSE; ///???

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
			const unsigned short *spWord = reinterpret_cast<const unsigned short*>(&dduWord[index-spWordCount-1]);
			if( nRecords<12 ) {
				if( sp[nRecords++].unpack(spWord) ) coruptions |= NONSENSE;
			} else {
				coruptions |= CONFIGURATION;
				break;
			}
		}

		index++;
	}

	return coruptions;
}
