/****************************************************************************
*
* This is a part of the TOTEM offline software.
* Authors:
*  Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#include "EventFilter/TotemRawToDigi/interface/SRSFileReader.h"

#include "EventFilter/TotemRawToDigi/interface/event_3_14.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include <cmath>

//----------------------------------------------------------------------------------------------------

using namespace std;

//----------------------------------------------------------------------------------------------------

const unsigned int SRSFileReader::eventHeaderSize = sizeof(eventHeaderStruct);

//----------------------------------------------------------------------------------------------------

SRSFileReader::SRSFileReader() : dataPtr(NULL), dataPtrSize(0), infile(NULL)
{
}

//----------------------------------------------------------------------------------------------------

SRSFileReader::~SRSFileReader()
{
  Close();

  if (dataPtr != NULL)
    delete [] dataPtr;
}

//----------------------------------------------------------------------------------------------------

int SRSFileReader::Open(const std::string &fn)
{
  infile = fopen(fn.c_str(), "r");

  if (infile == NULL)
  {
    perror("Error while opening file in SRSFileReader::Open");
    return 1;
  }

  return 0;
}

//----------------------------------------------------------------------------------------------------

void SRSFileReader::Close()
{
  if (infile)
    fclose(infile);

  infile = NULL;
}

//----------------------------------------------------------------------------------------------------

unsigned char SRSFileReader::ReadToBuffer(unsigned int bytesToRead, unsigned int offset)
{
#ifdef DEBUG
  printf(">> SRSFileReader::ReadToBuffer(%u, %u), this = %p, dataPtr = %p, dataPtrSize = %u\n",
      bytesToRead, offset, (void*) this, dataPtr, dataPtrSize);
#endif

  // allocate new memory block if current one is too small
  if (dataPtrSize < bytesToRead+offset)
  {
    char *newPtr = NULL;
    try {
      newPtr = new char[bytesToRead+offset];
    }
    catch (bad_alloc& ba)
    {
      cerr << "Error in SRSFileReader::ReadToBuffer > " << "Cannot allocate buffer large enough." << endl;
      return 2;
    }

    if (dataPtr)
    {
      if (offset > 0)
        memcpy(newPtr, dataPtr, offset);
      delete [] dataPtr;
    }

    dataPtr = newPtr;
    dataPtrSize = bytesToRead+offset;
  }

  // read data at given offset
  unsigned int bytesRead = fread(dataPtr + offset, sizeof(char), bytesToRead, infile);
  int eofFlag = feof(infile);

  if (bytesRead != bytesToRead && !(bytesRead == 0 && eofFlag)) 
  {
    cerr << "Error in SRSFileReader::ReadToBuffer > " << "Reading from file to buffer failed. Only " << bytesRead
  			      << " B read from " << bytesToRead << " B." << endl;
    return 1;
  }

  return 0;
}
//----------------------------------------------------------------------------------------------------

unsigned char SRSFileReader::GetNextEvent(TotemRawEvent &rawEvent, FEDRawDataCollection &dataColl)
{
#ifdef DEBUG
  printf(">> SRSFileReader::GetNextEvent, this = %p\n", (void*)this);
  printf("\teventHeaderSize = %u\n", eventHeaderSize);
#endif

  eventHeaderStruct *eventHeader = NULL;

  while (!feof(infile))
  {
    // read next header
    if (ReadToBuffer(eventHeaderSize, 0) != 0)
      return 10;

    eventHeader = (eventHeaderStruct *) dataPtr;

    // check the sanity of header data
    if (eventHeader->eventMagic != EVENT_MAGIC_NUMBER)
    {
      cerr << "Error in SRSFileReader::GetNextEvent > " << "Event magic check failed (" << hex
        << eventHeader->eventMagic << "!=" << EVENT_MAGIC_NUMBER << dec << "). Exiting." << endl;
      return 1;
    }

    unsigned int N = eventHeader->eventSize;
    if (N < eventHeaderSize)
    {
      cerr << "Error in SRSFileReader::GetNextEvent > " << "Event size (" << N
        << ") smaller than header size (" << eventHeaderSize << "). Exiting." << endl;
      return 1;
    }

    // get next event from the file (the header has already been read)
    if (ReadToBuffer(N-eventHeaderSize, eventHeaderSize) != 0)
      return 10;

    // because dataPtr could move, we have to be sure, that eventHeader still points the same as dataPtr
    eventHeader = (eventHeaderStruct *) dataPtr;

    // skip non physics events
    if (eventHeader->eventType != PHYSICS_EVENT)
      continue;

    break;
  }

  // check if the end of the file has been reached
  if (feof(infile))
    return 1;

  // process the buffer
  unsigned int errorCounter = ProcessDATESuperEvent(dataPtr, rawEvent, dataColl);

#ifdef DEBUG
  printf("* %u, %u, %u\n",
    EVENT_ID_GET_NB_IN_RUN( eventHeader->eventId ),
    EVENT_ID_GET_BURST_NB( eventHeader->eventId ),
    EVENT_ID_GET_NB_IN_BURST( eventHeader->eventId )
    );
#endif

  if (errorCounter > 0)
  {
    cerr << "Error in SRSFileReader::GetNextEvent > " << errorCounter << " GOH blocks have failed consistency checks in event "
      << rawEvent.dataEventNumber << "." << endl;
  }

  return 0;
}

//----------------------------------------------------------------------------------------------------

unsigned int SRSFileReader::ProcessDATESuperEvent(char *ptr, TotemRawEvent &rawEvent, FEDRawDataCollection &dataColl)
{
  eventHeaderStruct *eventHeader = (eventHeaderStruct *) ptr;
  bool superEvent = TEST_ANY_ATTRIBUTE(eventHeader->eventTypeAttribute, ATTR_SUPER_EVENT);

#ifdef DEBUG
  printf(">> SRSFileReader::ProcessVMEBEvent\n");

  printf("\teventSize = %i\n", eventHeader->eventSize);
  printf("\teventMagic = %i\n", eventHeader->eventMagic);
  printf("\teventHeadSize = %i\n", eventHeader->eventHeadSize);
  printf("\t* eventHeadSize (extra) = %i\n", eventHeader->eventHeadSize - EVENT_HEAD_BASE_SIZE);
  printf("\t* eventPayloadSize = %i\n", eventHeader->eventSize - eventHeader->eventHeadSize);
  printf("\teventVersion = %i\n", eventHeader->eventVersion);
  printf("\teventType = %i\n", eventHeader->eventType);
  printf("\teventRunNb = %i\n", eventHeader->eventRunNb);
  printf("\teventId = %p\n", (void*)eventHeader->eventId);
  printf("\teventTriggerPattern = %p\n", (void*)eventHeader->eventTriggerPattern);
  printf("\teventDetectorPattern = %p\n",(void*) eventHeader->eventDetectorPattern);
  printf("\teventTypeAttribute = %p\n", (void*)eventHeader->eventTypeAttribute);
  printf("\t* super event = %i\n", superEvent);
  printf("\teventLdcId = %i\n", eventHeader->eventLdcId);
  printf("\teventGdcId = %i\n", eventHeader->eventGdcId);
  printf("\teventTimestamp = %i\n", eventHeader->eventTimestamp);
#endif

  // store important GDC data
  rawEvent.dataEventNumber = EVENT_ID_GET_NB_IN_RUN(eventHeader->eventId) - 1;
  rawEvent.timestamp = eventHeader->eventTimestamp;

  eventSizeType eventSize = eventHeader->eventSize;
  eventHeadSizeType headSize = eventHeader->eventHeadSize;

  // process all sub-events (LDC frames)
  fedIdx = 0;
  unsigned int errorCounter = 0;
  if (superEvent)
  {
    unsigned int offset = headSize;
    while (offset < eventSize)
    {
#ifdef DEBUG 
      printf("\t> offset before %i\n", offset);
#endif
      eventStruct *subEvPtr = (eventStruct *) (ptr + offset); 
      eventSizeType subEvSize = subEvPtr->eventHeader.eventSize;

      errorCounter += ProcessDATEEvent(ptr + offset, rawEvent, dataColl);

      offset += subEvSize;
#ifdef DEBUG 
      printf("\t> offset after %i\n", offset);
#endif
    }
  } else
    errorCounter += ProcessDATEEvent(ptr, rawEvent, dataColl);

  return errorCounter;
}

//----------------------------------------------------------------------------------------------------

unsigned int SRSFileReader::ProcessDATEEvent(char *ptr, TotemRawEvent &rawEvent, FEDRawDataCollection &dataColl)
{
  eventHeaderStruct *eventHeader = (eventHeaderStruct *) ptr;

#ifdef DEBUG 
  printf("\t\t>> ProcessDATEEvent\n");

  printf("\t\teventSize = %u\n", eventHeader->eventSize);
  printf("\t\teventMagic = %u\n", eventHeader->eventMagic);
  printf("\t\teventHeadSize = %u\n", eventHeader->eventHeadSize);
  printf("\t\t* eventHeadSize (extra) = %u\n", eventHeader->eventHeadSize - EVENT_HEAD_BASE_SIZE);
  printf("\t\t* eventPayloadSize = %u\n", eventHeader->eventSize - eventHeader->eventHeadSize);
  printf("\t\teventVersion = %u\n", eventHeader->eventVersion);
  printf("\t\teventType = %u\n", eventHeader->eventType);
  printf("\t\teventRunNb = %u\n", eventHeader->eventRunNb);
  printf("\t\teventId = %p\n", (void*) eventHeader->eventId);
  printf("\t\teventTriggerPattern = %p\n", (void*) eventHeader->eventTriggerPattern);
  printf("\t\teventDetectorPattern = %p\n", (void*) eventHeader->eventDetectorPattern);
  printf("\t\teventTypeAttribute = %p\n", (void*) eventHeader->eventTypeAttribute);
  printf("\t\teventLdcId = %u\n", eventHeader->eventLdcId);
  printf("\t\teventGdcId = %u\n", eventHeader->eventGdcId);
  printf("\t\teventTimestamp = %u\n", eventHeader->eventTimestamp);
#endif

  // store important LDC data
  rawEvent.ldcTimeStamps[eventHeader->eventLdcId] = eventHeader->eventTimestamp;  

  unsigned long subEvSize = eventHeader->eventSize;
  unsigned long offset = eventHeader->eventHeadSize;

  // process all equipments (OptoRx frames)
  unsigned int errorCounter = 0;
  while (offset < subEvSize)
  {
#ifdef DEBUG 
    printf("\t\toffset (before) %lu\n", offset);
#endif
    
    equipmentHeaderStruct *eq = (equipmentHeaderStruct *) (ptr + offset);
    equipmentSizeType equipmentHeaderStructSize = sizeof(equipmentHeaderStruct);
    unsigned int payloadSize = eq->equipmentSize - equipmentHeaderStructSize;

    // check for presence of the "0xFAFAFAFA" word (32 bits)
    uint64_t *payloadPtr = (uint64_t *)(ptr + offset + equipmentHeaderStructSize);
    if ((*payloadPtr & 0xFFFFFFFF) == 0xFAFAFAFA)
    {
      payloadPtr = (uint64_t *)(ptr + offset + equipmentHeaderStructSize + 4);
      payloadSize -= 4;
    }

#ifdef DEBUG 
    printf("\t\t\tequipmentSize = %u\n", eq->equipmentSize);
    printf("\t\t\tequipmentType = %u\n", eq->equipmentType);
    printf("\t\t\tequipmentId = %u\n", eq->equipmentId);
    printf("\t\t\tequipmentTypeAttribute = %p\n", (void*) eq->equipmentTypeAttribute);
    printf("\t\t\tequipmentBasicElementSize = %u\n", eq->equipmentBasicElementSize);
    printf("\t\t\t\t\tpayload size = %u\n", payloadSize);
    printf("\t\t\t\t\tpayload ptr = %p\n", (void*) payloadPtr);
#endif

    if (payloadSize > 0)
    {
      switch (eq->equipmentType)
      {
        case etOptoRxVME:
        case etOptoRxSRS:
            MakeFEDRawData(payloadPtr, payloadSize, dataColl);
          break;

        default:
          cerr << "Error in SRSFileReader::ProcessDATEEvent > " << "Unknown equipment type: " << eq->equipmentType << ". Skipping." << endl;
      }
    } 

    offset += eq->equipmentSize;

#ifdef DEBUG 
    printf("\t\toffset (after) %lu\n", offset);
#endif
  }

  return errorCounter;
}

//----------------------------------------------------------------------------------------------------

void SRSFileReader::MakeFEDRawData(uint64_t *payloadPtr, unsigned int payloadSize, FEDRawDataCollection &dataColl)
{
  FEDRawData &rd = dataColl.FEDData(fedIdx++);

  unsigned int fedSize = payloadSize;
  rd.resize(fedSize);

  unsigned char *buffer = rd.data();
  memcpy(buffer, payloadPtr, payloadSize);
}
