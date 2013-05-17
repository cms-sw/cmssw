#include "FileReaderDCC.h"
#include <iostream>    // cerr
#include <errno.h>     // errno
#include <string.h>    // bzero, memcpy
#include <stdlib.h>    // exit
#include <sys/types.h> // open
#include <sys/stat.h>  // open
#include <fcntl.h>     // open
#include <unistd.h>    // read, close

#ifndef O_LARGEFILE //for OSX
#define O_LARGEFILE 0
#endif

FileReaderDCC::FileReaderDCC(void){
	if( sizeof(unsigned long long)!=8 || sizeof(unsigned short)!=2 )
		throw std::runtime_error(std::string("Wrong platform: sizeof(unsigned long long)!=8 || sizeof(unsigned short)!=2"));
	raw_event = new unsigned short [200000*40];
	end = (file_buffer_end = file_buffer + sizeof(file_buffer)/sizeof(unsigned long long));
	bzero(raw_event,  sizeof(raw_event)  );
	bzero(file_buffer,sizeof(file_buffer));
	word_0=0; word_1=0; word_2=0;
	eventStatus = 0;
	selectCriteria = Header|Trailer;
	rejectCriteria = DCCoversize|FFFF|Unknown;
	acceptCriteria = 0x3F; // Everything
	fd = 0;
}

FileReaderDCC::~FileReaderDCC(void){ if( fd ) close(fd); }

int FileReaderDCC::open(const char *filename) throw (std::runtime_error) {
	if( fd ) close(fd);
	fd = ::open(filename,O_RDONLY|O_LARGEFILE);
	if( fd == -1 ) throw ( std::runtime_error(std::string("Error opening ").append(filename).append(" data file.")) );
	return fd;
}

size_t FileReaderDCC::read(const unsigned short* &buf) throw (std::runtime_error) {
	// Check for ubnormal situation
	if( end>file_buffer_end || end<file_buffer ) throw ( std::runtime_error("Error of reading") );
	if( !fd ) throw ( std::runtime_error("Open some file first") );

	unsigned long long *start = end;
	unsigned short     *event = raw_event;

	eventStatus = 0;
	size_t dccWordCount = 0;
	end = 0;

	while( !end && dccWordCount<50000*40 ){
		unsigned long long *dccWord = start;
		unsigned long long preHeader = 0;

		// Did we reach end of current buffer and want to read next block?
		// If it was first time and we don't have file buffer then we won't get inside
		while( dccWord<file_buffer_end && dccWordCount<50000 ){
			word_0 =  word_1; // delay by 2 DCC words
			word_1 =  word_2; // delay by 1 DCC word
			word_2 = *dccWord;// current DCC word
			if( (word_1&0xF0000000000000FFLL)==0x500000000000005FLL && // let's call this a preHeader
				(word_2&0xFF000000000000FFLL)==0xD900000000000017LL ){ // and this is a header
				if( eventStatus&Header ){ // Second header
					word_2 = word_1; // Fall back to get rigth preHeader next time
					end = dccWord;   // Even if we end with preHeader of next evet put it to the end of this event too
					break;
				}
				if( dccWordCount>1 ){ // Extra words between trailer and header
					if( (word_0&0xFFFFFFFFFFFF0000LL)==0xFFFFFFFFFFFF0000LL ) eventStatus |= FFFF;
					word_2 = word_1; // Fall back to get rigth preHeader next time
					end = dccWord;
					break;
				}
				eventStatus |= Header;
				if( event==raw_event ) preHeader = word_1; // If preHeader not yet in event then put it there
				start = dccWord;
			}
			if( (word_1&0xFF00000000000000LL)==0xEF00000000000000LL &&
				(word_2&0xFF0000000000000FLL)==0xAF00000000000007LL ){
				eventStatus |= Trailer;
				end = ++dccWord;
				break;
			}
			// Increase counters by one DCC word
			dccWord++;
			dccWordCount++;
		}

		// If have DCC Header
		if( preHeader ){
			// Need to account first word of DCC Header
			memcpy(event,&preHeader,sizeof(preHeader));
			event += sizeof(preHeader)/sizeof(unsigned short);
		}

		// Take care of the rest
		memcpy(event,start,(dccWord-start)*sizeof(unsigned long long));
		event += (dccWord-start)*sizeof(unsigned long long)/sizeof(unsigned short);

		// If reach max length
		if( dccWordCount==50000*40 ){ end = dccWord; break; }

		if( !end ){
			// Need to read next block for the rest of this event
			ssize_t length = ::read(fd,file_buffer,sizeof(file_buffer));
			if( length==-1 ) throw ( std::runtime_error("Error of reading") );
			if( length== 0 ){
				eventStatus |= EndOfStream;
				end = (file_buffer_end = file_buffer + sizeof(file_buffer)/sizeof(unsigned long long));
				break;
			}
			file_buffer_end = file_buffer + length/sizeof(unsigned long long);

			// Will start from the beginning of new buffer next time we read it
			start = file_buffer;
		}
	}

	if( !end ) eventStatus |= DCCoversize;
	if( !(eventStatus&Header) && !(eventStatus&Trailer) && !(eventStatus&FFFF) ) eventStatus |= Unknown;

	buf = (const unsigned short*)raw_event;
	return (eventStatus&FFFF?event-raw_event-4:event-raw_event);
}

size_t FileReaderDCC::next(const unsigned short* &buf) throw(std::runtime_error) {
	size_t size=0;
	do {
		if( (size = read(buf)) == 0 ) break;
	} while( rejectCriteria&eventStatus || !(acceptCriteria&eventStatus) || (selectCriteria?selectCriteria!=eventStatus:0) );
	return size;
}
