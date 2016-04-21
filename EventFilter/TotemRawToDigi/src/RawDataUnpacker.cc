/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#include "EventFilter/TotemRawToDigi/interface/RawDataUnpacker.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

RawDataUnpacker::RawDataUnpacker(const edm::ParameterSet &conf)
{
}

//----------------------------------------------------------------------------------------------------

int RawDataUnpacker::Run(int fedId, const FEDRawData &data, vector<TotemFEDInfo> &fedInfoColl, SimpleVFATFrameCollection &coll)
{
  unsigned int size_in_words = data.size() / 8; // bytes -> words
  if (size_in_words < 2)
  {
    LogProblem("Totem") << "Error in RawDataUnpacker::Run > " <<
      "Data in FED " << fedId << " too short (size = " << size_in_words << " words).";
    return 1;
  }

  fedInfoColl.push_back(TotemFEDInfo(fedId));

  return ProcessOptoRxFrame((const word *) data.data(), size_in_words, fedInfoColl.back(), &coll);
}

//----------------------------------------------------------------------------------------------------

int RawDataUnpacker::ProcessOptoRxFrame(const word *buf, unsigned int frameSize, TotemFEDInfo &fedInfo, SimpleVFATFrameCollection *fc)
{
  // get OptoRx metadata
  unsigned long long head = buf[0];
  unsigned long long foot = buf[frameSize-1];

  fedInfo.setHeader(head);
  fedInfo.setFooter(foot);

  unsigned int BOE = (head >> 60) & 0xF;
  unsigned int H0 = (head >> 0) & 0xF;

  //unsigned long LV1 = (head >> 32) & 0xFFFFFF;
  //unsigned long BX = (head >> 20) & 0xFFF;
  unsigned int OptoRxId = (head >> 8) & 0xFFF;
  unsigned int FOV = (head >> 4) & 0xF;

  unsigned int EOE = (foot >> 60) & 0xF;
  unsigned int F0 = (foot >> 0) & 0xF;
  unsigned int FSize = (foot >> 32) & 0x3FF;

  // check header and footer structure
  if (BOE != 5 || H0 != 0 || EOE != 10 || F0 != 0 || FSize != frameSize)
  {
    LogProblem("Totem") << "Error in RawDataUnpacker::ProcessOptoRxFrame > " << "Wrong structure of OptoRx header/footer: "
      << "BOE=" << BOE << ", H0=" << H0 << ", EOE=" << EOE << ", F0=" << F0
      << ", size (OptoRx)=" << FSize << ", size (DATE)=" << frameSize
      << ". OptoRxID=" << OptoRxId << ". Skipping frame." << endl;
    return 0;
  }

  #ifdef DEBUG
    printf(">> RawDataUnpacker::ProcessOptoRxFrame > OptoRxId = %u, BX = %lu, LV1 = %lu, frameSize = %u, subFrames = %u)\n",
      OptoRxId, BX, LV1, frameSize, subFrames);
  #endif

  // parallel or serial transmission?
  if (FOV == 1)
    return ProcessOptoRxFrameSerial(buf, frameSize, fc);

  if (FOV == 2)
    return ProcessOptoRxFrameParallel(buf, frameSize, fedInfo, fc);

  LogProblem("Totem") << "Error in RawDataUnpacker::ProcessOptoRxFrame > " << "Unknown FOV = " << FOV << endl;

  return 0;
}

//----------------------------------------------------------------------------------------------------

int RawDataUnpacker::ProcessOptoRxFrameSerial(const word *buf, unsigned int frameSize, SimpleVFATFrameCollection *fc)
{
  // get OptoRx metadata
  unsigned int OptoRxId = (buf[0] >> 8) & 0xFFF;

  // get number of subframes
  unsigned int subFrames = (frameSize - 2) / 194;

  // process all sub-frames
  unsigned int errorCounter = 0;
  for (unsigned int r = 0; r < subFrames; ++r)
  {
    for (unsigned int c = 0; c < 4; ++c)
    {
      unsigned int head = (buf[1 + 194 * r] >> (16 * c)) & 0xFFFF;
      unsigned int foot = (buf[194 + 194 * r] >> (16 * c)) & 0xFFFF;

      #ifdef DEBUG
        printf(">>>> r = %i, c = %i: S = %i, BOF = %i, EOF = %i, ID = %i, ID' = %i\n", r, c, head & 0x1, head >> 12, foot >> 12, (head >> 8) & 0xF, (foot >> 8) & 0xF);
      #endif

      // stop if this GOH is NOT active
      if ((head & 0x1) == 0)
        continue;

      #ifdef DEBUG
        printf("\tHeader active (%04x -> %x).\n", head, head & 0x1);
      #endif

      // check structure
      if (head >> 12 != 0x4 || foot >> 12 != 0xB || ((head >> 8) & 0xF) != ((foot >> 8) & 0xF))
      {
        char ss[500];
        if (head >> 12 != 0x4)
          sprintf(ss, "\n\tHeader is not 0x4 as expected (%x).", head);
        if (foot >> 12 != 0xB)
          sprintf(ss, "\n\tFooter is not 0xB as expected (%x).", foot);
        if (((head >> 8) & 0xF) != ((foot >> 8) & 0xF))
          sprintf(ss, "\n\tIncompatible GOH IDs in header (%x) and footer (%x).", ((head >> 8) & 0xF),
            ((foot >> 8) & 0xF));

        LogProblem("Totem") << "Error in RawDataUnpacker::ProcessOptoRxFrame > " << "Wrong payload structure (in GOH block row " << r <<
          " and column " << c << ") in OptoRx frame ID " << OptoRxId << ". GOH block omitted." << ss << endl;

        errorCounter++;
        continue;
      }

      // allocate memory for VFAT frames
      unsigned int goh = (head >> 8) & 0xF;
      vector<VFATFrame::word*> dataPtrs;
      for (unsigned int fi = 0; fi < 16; fi++)
      {
        TotemFramePosition fp(0, 0, OptoRxId, goh, fi);
        dataPtrs.push_back( fc->InsertEmptyFrame(fp)->getData() );
      }

      #ifdef DEBUG
        printf(">>>> transposing GOH block at prefix: %i, dataPtrs = %p\n", OptoRxId*192 + goh*16, dataPtrs);
      #endif

      // deserialization
      for (int i = 0; i < 192; i++)
      {
        int iword = 11 - i / 16;  // number of current word (11...0)
        int ibit = 15 - i % 16;   // number of current bit (15...0)
        unsigned int w = (buf[i + 2 + 194 * r] >> (16 * c)) & 0xFFFF;

        // Fill the current bit of the current word of all VFAT frames
        for (int idx = 0; idx < 16; idx++)
        {
          if (w & (1 << idx))
            dataPtrs[idx][iword] |= (1 << ibit);
        }
      }
    }
  }

  return errorCounter;
}

//----------------------------------------------------------------------------------------------------

int RawDataUnpacker::ProcessOptoRxFrameParallel(const word *buf, unsigned int frameSize, TotemFEDInfo &fedInfo, SimpleVFATFrameCollection *fc)
{
  // get OptoRx metadata
  unsigned long long head = buf[0];
  unsigned int OptoRxId = (head >> 8) & 0xFFF;

  // recast data as buffer or 16bit words, skip header
  const uint16_t *payload = (const uint16_t *) (buf + 1);

  // read in OrbitCounter block
  const uint32_t *ocPtr = (const uint32_t *) payload;
  fedInfo.setOrbitCounter(*ocPtr);
  payload += 2;

  // size in 16bit words, without header, footer and orbit counter block
  unsigned int nWords = (frameSize-2) * 4 - 2;

  // process all VFAT data
  for (unsigned int offset = 0; offset < nWords;)
  {
    unsigned int wordsProcessed = ProcessVFATDataParallel(payload + offset, OptoRxId, fc);
    offset += wordsProcessed;
  }

  return 0;
}

//----------------------------------------------------------------------------------------------------

int RawDataUnpacker::ProcessVFATDataParallel(const uint16_t *buf, unsigned int OptoRxId, SimpleVFATFrameCollection *fc)
{
  // start counting processed words
  unsigned int wordsProcessed = 1;

  // padding word? skip it
  if (buf[0] == 0xFFFF)
    return wordsProcessed;

  // check header flag
  unsigned int hFlag = (buf[0] >> 8) & 0xFF;
  if (hFlag != vmCluster && hFlag != vmRaw)
  {
    LogProblem("Totem") << "Error in RawDataUnpacker::ProcessVFATDataParallel > "
      << "Unknown header flag " << hFlag << ". Skipping this word." << endl;
    return wordsProcessed;
  }

  // compile frame position
  // NOTE: DAQ group uses terms GOH and fiber in the other way
  unsigned int gohIdx = (buf[0] >> 4) & 0xF;
  unsigned int fiberIdx = (buf[0] >> 0) & 0xF;
  TotemFramePosition fp(0, 0, OptoRxId, gohIdx, fiberIdx);

  // prepare temporary VFAT frame
  VFATFrame f;
  VFATFrame::word *fd = f.getData();

  // copy footprint, BC, EC, Flags, ID, if they exist
  f.presenceFlags = 0;

  if (((buf[wordsProcessed] >> 12) & 0xF) == 0xA)  // BC
  {
    f.presenceFlags |= 0x1;
    fd[11] = buf[wordsProcessed];
    wordsProcessed++;
  }

  if (((buf[wordsProcessed] >> 12) & 0xF) == 0xC)  // EC, flags
  {
    f.presenceFlags |= 0x2;
    fd[10] = buf[wordsProcessed];
    wordsProcessed++;
  }

  if (((buf[wordsProcessed] >> 12) & 0xF) == 0xE)  // ID
  {
    f.presenceFlags |= 0x4;
    fd[9] = buf[wordsProcessed];
    wordsProcessed++;
  }

  // save offset where channel data start
  unsigned int dataOffset = wordsProcessed;

  // find trailer
  if (hFlag == vmCluster)
  {
    unsigned int nCl = 0;
    while ( (buf[wordsProcessed + nCl] >> 12) != 0xF )
      nCl++;

    wordsProcessed += nCl;
  }

  if (hFlag == vmRaw)
    wordsProcessed += 9;

  // process trailer
  unsigned int tSig = buf[wordsProcessed] >> 12;
  unsigned int tErrFlags = (buf[wordsProcessed] >> 8) & 0xF;
  unsigned int tSize = buf[wordsProcessed] & 0xFF;

  f.daqErrorFlags = tErrFlags;

  bool skipFrame = false;
  bool suppressChannelErrors = false;

  if (tSig != 0xF)
  {
    LogProblem("Totem") << "Error in RawDataUnpacker::ProcessVFATDataParallel > "
      << "Wrong trailer signature (" << tSig << ") at "
      << fp << ". This frame will be skipped." << endl;
    skipFrame = true;
  }

  if (tErrFlags != 0)
  {
    LogProblem("Totem") << "Error in RawDataUnpacker::ProcessVFATDataParallel > "
      << "Error flags not zero (" << tErrFlags << ") at "
      << fp << ". Channel errors will be suppressed." << endl;
    suppressChannelErrors = true;
  }

  wordsProcessed++;

  if (tSize != wordsProcessed)
  {
    LogProblem("Totem") << "Error in RawDataUnpacker::ProcessVFATDataParallel > "
      << "Trailer size (" << tSize << ") does not match with words processed ("
      << wordsProcessed << ") at " << fp << ". This frame will be skipped." << endl;
    skipFrame = true;
  }

  if (skipFrame)
    return wordsProcessed;

  // get channel data - cluster mode
  if (hFlag == vmCluster)
  {
    unsigned int nCl = 0;
    while ( (buf[dataOffset + nCl] >> 12) != 0xF )
    {
      const uint16_t &w = buf[dataOffset + nCl];
      unsigned int clSize = (w >> 8) & 0x7F;
      unsigned int clPos = (w >> 0) & 0xFF;

      // special case: size 0 means chip full
      if (clSize == 0)
        clSize = 128;

      nCl++;

      // activate channels
      //  convention - range <pos, pos-size+1>
      signed int chMax = clPos;
      signed int chMin = clPos - clSize + 1;
      if (chMax < 0 || chMax > 127 || chMin < 0 || chMin > 127 || chMin > chMax)
      {
        if (!suppressChannelErrors)
          LogProblem("Totem") << "Error in RawDataUnpacker::ProcessVFATDataParallel > "
            << "Invalid cluster (pos=" << clPos
            << ", size=" << clSize << ", min=" << chMin << ", max=" << chMax << ") at " << fp
            <<". Skipping this cluster." << endl;

        continue;
      }

      for (signed int ch = chMin; ch <= chMax; ch++)
      {
        unsigned int wi = ch / 16;
        unsigned int bi = ch % 16;
        fd[wi + 1] |= (1 << bi);
      }
    }
  }

  // get channel data and CRC - raw mode
  if (hFlag == vmRaw)
  {
    for (unsigned int i = 0; i < 8; i++)
      fd[8 - i] = buf[dataOffset + i];

    // copy CRC
    f.presenceFlags |= 0x8;
    fd[0] = buf[dataOffset + 8];
  }

  // save frame to output
  fc->Insert(fp, f);

  return wordsProcessed;
}
