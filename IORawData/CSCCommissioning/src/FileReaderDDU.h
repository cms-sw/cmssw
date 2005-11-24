// Author: Khristian Kotov ( University of Florida DQM group )
#ifndef FileReaderDDU_h
#define FileReaderDDU_h

#include <stdexcept>   // std::runtime_error
#include <unistd.h>    // size_t

//#ifdef WITHOUT_DDUREADER
class FileReaderDDU {
//#else
//#include "DDUReader.h"
//class FileReaderDDU : public DDUReader {
//#endif
private:
	unsigned short raw_event[200000];

	unsigned long long word_0, word_1, word_2; // To remember some history
	unsigned long long file_buffer[4000];      // Read data block for efficiency

	unsigned long long *end, *file_buffer_end; // where stoped last time and where is end

public:
	enum {Header=1,Trailer=2,DDUoversize=4,FFFF=8,Unknown=16,EndOfStream=32};
	enum {Type1=Header|Trailer, Type2=Header, Type3=Header|DDUoversize, Type4=Trailer, Type5=Unknown, Type6=Unknown|DDUoversize, Type7=FFFF}; // Andrey Korytov's notations
private:
	unsigned int eventStatus, selectCriteria, acceptCriteria, rejectCriteria;

	int fd;

public:
	int    open(const char *filename) throw (std::runtime_error);
	size_t read(const unsigned short* &buf) throw (std::runtime_error); // Just plain read function
	size_t next(const unsigned short* &buf) throw (std::runtime_error); // Same as ``read'', but returns only events pass certain criteria
	void select(unsigned int criteria) throw() { selectCriteria = criteria; } // return events satisfying all criteria
	void accept(unsigned int criteria) throw() { acceptCriteria = criteria; } // return all events satisfying any of criteria
	void reject(unsigned int criteria) throw() { rejectCriteria = criteria; } // return events not satisfying any of criteria

	unsigned int status(void) const throw() { return eventStatus; }

//#ifndef WITHOUT_DDUREADER
//	int openFile(std::string filename){ fd_schar = open(filename.c_str()); return 0; }
//	int readDDU(unsigned short **buf, const bool debug = false){ const unsigned short *qqq; int len = 2*next(qqq); *buf = const_cast<unsigned short*>(qqq); return len; }
//
//	virtual int chunkSize(void)    {return 0;} // Needed for EmuRUI
//	virtual int reset    (void)    {return 0;} // Needed for EmuRUI
//	virtual int enableBlock (void) {return 0;} // Needed for EmuRUI
//	virtual int disableBlock(void) {return 0;} // Needed for EmuRUI
//	virtual int endBlockRead(void) {return 0;} // Needed for EmuRUI
//#endif

	FileReaderDDU(void);
	virtual ~FileReaderDDU(void);
};

#endif
