/****************************************************************************
*
* This is a part of the TOTEM offline software.
* Authors:
*  Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#ifndef _SRSFileReader_h_
#define _SRSFileReader_h_

#include "EventFilter/TotemRawToDigi/interface/SimpleVFATFrameCollection.h"

#include "DataFormats/TotemRawData/interface/TotemRawEvent.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include <vector>
#include <cstdio>

//----------------------------------------------------------------------------------------------------

/**
 * Reads a raw-data file in SRS format.
 **/ 
class SRSFileReader
{
public:
    typedef uint64_t word;

    /// standard equipment types
    enum { etOptoRxVME = 120, etOptoRxSRS = 22 };

    SRSFileReader();

    virtual ~SRSFileReader();

    virtual int Open(const std::string &);

    virtual void Close();

    virtual unsigned char GetNextEvent(TotemRawEvent &rawEvent, FEDRawDataCollection &);

protected:
    static const unsigned int eventHeaderSize;

    /// Processes one DATE super-event (GDC).
    /// returns the number of GOH blocks that failed consistency checks
    unsigned int ProcessDATESuperEvent(char *ptr, TotemRawEvent &rawEvent, FEDRawDataCollection &dataColl);

    /// Processes one DATE event (LDC).
    /// returns the number of GOH blocks that failed consistency checks
    unsigned int ProcessDATEEvent(char *ptr, TotemRawEvent &rawEvent, FEDRawDataCollection &dataColl);

    /// reads 'bytesToRead' bytes from the file to buffer, starting at the given offset
    virtual unsigned char ReadToBuffer(unsigned int bytesToRead, unsigned int offset);

    /// Inserts FEDRawData for each OptoRx.
    void MakeFEDRawData(uint64_t *payloadPtr, unsigned int payloadSize, FEDRawDataCollection &dataColl);

    /// data pointer, to be allocated one time only
    char *dataPtr;
 
 	/// data buffer size
    unsigned int dataPtrSize;
 
 	/// input file pointer
 	FILE *infile;

    /// Index for FEDRawDataCollection
    unsigned int fedIdx;
};

#endif
