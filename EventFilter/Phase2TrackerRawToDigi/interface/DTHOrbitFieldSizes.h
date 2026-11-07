#ifndef DTHOrbitFieldSizes_H
#define DTHOrbitFieldSizes_H

// Constants for bit field widths, markers, and size in BYTES
// From EDMS document No.: 2705647 v 2.2
// https://edms.cern.ch/document/2705647/2.2
constexpr unsigned int orbitHeaderSize = 32;
constexpr unsigned int fragmentTrailerSize = 16;
constexpr unsigned int fragmentPayloadWordSize = 16;  // Each fragment payload word is 16 bytes
constexpr unsigned int orbitVersionSize = 2;          // The version field size is 2 bytes
constexpr unsigned int sourceIdSize = 4;              // The source ID  field is 4 bytes
constexpr unsigned int runNumberSize = 4;             // The run number field is 4 bytes
constexpr unsigned int orbitNumberSize = 4;           // The orbit number field is 4 bytes
constexpr unsigned int eventCountResSize = 4;         // The event count reserved field is 4 bytes
constexpr unsigned int packetWordCountSize = 4;       // The packet word count  field is 4 bytes
constexpr unsigned int flagsSize = 4;                 // The flag field is 4 bytes
constexpr unsigned int checksumSize = 4;              // The checksum field is 4 bytes
constexpr unsigned int fragFlagSize = 2;              // The fragment flag field is 2 bytes
constexpr unsigned int fragSizeSize = 4;              // The fragment size field is 4 bytes
constexpr uint8_t orbitHeaderMarkerH = 0x48;
constexpr uint8_t orbitHeaderMarkerO = 0x4F;
constexpr uint8_t fragmentTrailerMarkerT = 0x54;
constexpr uint8_t fragmentTrailerMarkerF = 0x46;

//eventID+res+CRC = 8 bytes, i.e. offset of 8 bytes
constexpr size_t trailerOffsetEventId = 8;
constexpr size_t eventIdSize = 6;  // 6 bytes, the eventId is only 44 bits (5.5 bytes) out of the 48 bits (6 bytes)
constexpr uint64_t eventIdMask =
    0xFFFFFFFFFFFULL;  // mask to keep the first 44 bits (5.5 bytes), which is the eventId, out of the 48 bits

constexpr size_t trailerOffsetCRC = trailerOffsetEventId + eventIdSize;
constexpr size_t crcSize = 2;
#endif
