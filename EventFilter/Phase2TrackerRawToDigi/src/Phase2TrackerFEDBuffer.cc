#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDBuffer.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#define EDM_ML_DEBUG // remove this !

namespace Phase2Tracker
{

  // implementation of Phase2TrackerFEDBuffer
  Phase2TrackerFEDBuffer::Phase2TrackerFEDBuffer(const uint8_t* fedBuffer, const size_t fedBufferSize) 
    : buffer_(fedBuffer),
      bufferSize_(fedBufferSize)
  {
      
      LogTrace("Phase2TrackerFEDBuffer") << "content of buffer with size: "<<int(fedBufferSize)<<std::endl;
      for ( size_t i = 0;  i < fedBufferSize; i += 8)
      {
        uint64_t word  = read64(i,buffer_);
        LogTrace("Phase2TrackerFEDBuffer") << " word " << std::setfill(' ') << std::setw(2) << i/8 << " | " 
        << std::hex << std::setw(16) << std::setfill('0') << word << std::dec << std::endl;
      }
      LogTrace("Phase2TrackerFEDBuffer") << std::endl;

      // reserve all channels to avoid vector reservation updates (should be 16x16 in our case)
      channels_.reserve(MAX_FE_PER_FED*MAX_CBC_PER_FE);
      // first 64 bits word is for DAQ header
      daqHeader_     = FEDDAQHeader(buffer_);
      // last 64 bit word is daq trailer
      daqTrailer_    = FEDDAQTrailer(buffer_+bufferSize_-8);
      // tracker header follows daq header
      trackerHeader_ = Phase2TrackerFEDHeader(buffer_+8);
      // get pointer to payload
      payloadPointer_ = getPointerToPayload(); 
      // fill list of Phase2TrackerFEDChannels and get pointers to trigger and comissioning data
      findChannels();
  }
  
  Phase2TrackerFEDBuffer::~Phase2TrackerFEDBuffer()
  {
  }
  
  void Phase2TrackerFEDBuffer::findChannels()
  {
    // each FED can be connectd to up to 72 frontends (read from header)
    // each fronted can be connected to up to 16 CBC
    // in unsparsified (raw) mode, a header of 16bits tells which CBC are activated on this FE
    // in sparsified (ZS) mode, the header tells how many P and S clusters to expect for this FE. Each cluster contains the CBC ID at the beginning
    // one channel corresponds to one CBC chip, undependently of the mode
  
    // offset of beginning of current channel
    size_t offsetBeginningOfChannel = 0;
  
    // iterate over all FEs to see if they are active
    std::vector<bool>::iterator FE_it;
    std::vector<bool> status = trackerHeader_.frontendStatus();
  
    if(readoutMode() == READOUT_MODE_PROC_RAW)
    {
      for (FE_it = status.begin(); FE_it < status.end(); FE_it++)
      {
        // if the current fronted is on, fill channels and advance pointer to end of channel 
        if(*FE_it)
        {
          // read first FEDCH_PER_FEUNIT bits to know which CBC are on
          uint16_t cbc_status = static_cast<uint16_t>(*(payloadPointer_ + (offsetBeginningOfChannel^7))<<8); 
          cbc_status         += static_cast<uint16_t>(*(payloadPointer_ + ((offsetBeginningOfChannel + 1)^7))); 

          // advance pointer by FEDCH_PER_FEUNIT bits
          offsetBeginningOfChannel += MAX_CBC_PER_FE/8;
          for (int i=0; i<MAX_CBC_PER_FE; i++) 
          {
            // if CBC is ON, fill channel and advance pointer. else, push back empty channel
            if((cbc_status>>i)&0x1)
            {
              // Warning: STRIPS_PADDING+STRIPS_PER_CBC should always be an entire number of bytes
              channels_.push_back(Phase2TrackerFEDChannel(payloadPointer_,offsetBeginningOfChannel,(STRIPS_PADDING+STRIPS_PER_CBC)/8));
              offsetBeginningOfChannel += (STRIPS_PADDING+STRIPS_PER_CBC)/8;
            }
            else
            {
              channels_.push_back(Phase2TrackerFEDChannel(NULL,0,0));
            }
          }
        }
        else
        {
          // else fill with FEDCH_PER_FEUNIT null channels, don't advance the channel pointer 
          channels_.insert(channels_.end(),size_t(MAX_CBC_PER_FE),Phase2TrackerFEDChannel(payloadPointer_,0,0));
        }
      }
    }
    else if (readoutMode() == READOUT_MODE_ZERO_SUPPRESSED)
    {
      // save current bit index
      int bitOffset = 0;
      // loop on FE
      for (FE_it = status.begin(); FE_it < status.end(); FE_it++)
      {
        if(*FE_it)
        {
          // Read FE header : 
          // FE header is : P/S (1bit) + #P clusters (5bits if P, 0 otherwise) + #S clusters (5bits)
          // read type of module (2S/PS)
          uint8_t mod_type = static_cast<uint8_t>(read_n_at_m(payloadPointer_,1,bitOffset));
          // note the index of the next channel to fill
          int ichan = channels_.size();
          // add the proper number of channels for this FE (2 channels per chip) 
          channels_.insert(channels_.end(),size_t(MAX_CBC_PER_FE*2),Phase2TrackerFEDChannel(payloadPointer_,0,0));
          // read number of clusters of each type
          uint8_t num_p, num_s;
          if (mod_type == 0)
          {
              num_p = 0; 
              num_s = static_cast<uint8_t>(read_n_at_m(payloadPointer_,5,bitOffset+1));
              bitOffset += 6;
          }
          else
          {
              num_p = static_cast<uint8_t>(read_n_at_m(payloadPointer_,5,bitOffset+1));
              num_s = static_cast<uint8_t>(read_n_at_m(payloadPointer_,5,bitOffset+6));
              bitOffset += 11;
          }
          // start indexing
          int currCBC = -1;
          int iCBC = -1;
          int chanSize = 0;
          for (int i=0; i<num_p; i++)
          {
              // test if new chip (CBC for 2S, ??? for PS)
              iCBC = static_cast<uint8_t>(read_n_at_m(payloadPointer_,4,bitOffset+14));
              if(iCBC != currCBC or i==num_p-1)
              {
                  if(currCBC >= 0) 
                  {
                      // save channel (P channels start after S channels)
                      channels_[ichan + iCBC + MAX_CBC_PER_FE] = Phase2TrackerFEDChannel(payloadPointer_,bitOffset/8,(chanSize + 8 -1)/8,bitOffset%8,DET_PonPS);
                  }
                  chanSize = P_CLUSTER_SIZE_BITS;
              }
              else 
              {
                  chanSize += P_CLUSTER_SIZE_BITS;
              }
              // advance bit pointer
              bitOffset += P_CLUSTER_SIZE_BITS;
              currCBC = iCBC;
          }
          currCBC = -1;
          for (int i=0; i<num_s; i++)
          {
              iCBC = static_cast<uint8_t>(read_n_at_m(payloadPointer_,4,bitOffset+11));
              if(iCBC != currCBC or i==num_s-1)
              {
                  if(currCBC >= 0) 
                  {
                      // save channel
                      DET_TYPE det_type = (mod_type == 0) ? DET_SonPS : DET_Son2S;
                      channels_[ichan + iCBC] = Phase2TrackerFEDChannel(payloadPointer_,bitOffset/8,(chanSize + 8 -1)/8,bitOffset%8,det_type);
                  }
                  chanSize = S_CLUSTER_SIZE_BITS;
              }
              else 
              {
                  chanSize += S_CLUSTER_SIZE_BITS;
              }
              // advance bit pointer
              bitOffset += S_CLUSTER_SIZE_BITS;
              currCBC = iCBC;
          }
        }
        else
        {
          // else fill with null channels, don't advance the channel pointer
          channels_.insert(channels_.end(),size_t(MAX_CBC_PER_FE),Phase2TrackerFEDChannel(payloadPointer_,0,0));
        }
      }
      // compute byte offset for payload
      offsetBeginningOfChannel = (bitOffset + 8 -1)/8;
    }
    // round the offset to the next 64 bits word
    int words64 = (offsetBeginningOfChannel + 8 - 1)/8; // size in 64 bit
    int payloadSize = words64 * 8; // size in bytes
    triggerPointer_ = payloadPointer_ + payloadSize;

    // get diff size in bytes:
    // fedBufferSize - (DAQHeader+TrackHeader+PayloadSize+TriggerSize+DAQTrailer)
    int bufferDiff = bufferSize_ - 8 - trackerHeader_.getTrackerHeaderSize()
                   - payloadSize - TRIGGER_SIZE - 8;

    // check if condition data is supposed to be there:
    if(trackerHeader_.getConditionData())
    {
      condDataPointer_  = triggerPointer_ + TRIGGER_SIZE;
      // diff must be equal to condition data size
      if (bufferDiff <= 0)
      {
        std::ostringstream ss;
        ss << "[Phase2Tracker::Phase2TrackerFEDBuffer::"<<__func__<<"] " << "\n";
        ss << "FED Buffer Size does not match data => missing condition data? : " << "\n";
        ss << "Expected Buffer Size " << bufferSize_ << " bytes" << "\n";
        ss << "Computed Buffer Size " << bufferSize_ + bufferDiff << " bytes" << "\n";
        throw cms::Exception("Phase2TrackerFEDBuffer") << ss.str();
      }
    }
    else
    {
      // put a null pointer to indicate lack of condition data
      condDataPointer_ = 0;
      // check buffer size :
      if (bufferDiff != 0)
      {
        std::ostringstream ss;
        ss << "[Phase2Tracker::Phase2TrackerFEDBuffer::"<<__func__<<"] " << "\n";
        ss << "FED Buffer Size does not match data => corrupted buffer? : " << "\n";
        ss << "Expected Buffer Size " << bufferSize_ << " bytes" << "\n";
        ss << "Computed Buffer Size " << bufferSize_ + bufferDiff << " bytes" << "\n";
        throw cms::Exception("Phase2TrackerFEDBuffer") << ss.str();
      }
    } 

  }
  
  std::map<uint32_t,uint32_t> Phase2TrackerFEDBuffer::conditionData() const
  {
      std::map<uint32_t,uint32_t> cdata;
      // check if there is condition data
      if(condDataPointer_)
      {
        const uint8_t* pointer = condDataPointer_;
        const uint8_t* stop    = buffer_ + bufferSize_ - 8;
        // first read the size
        uint32_t size = 0;
        // somehow the size is not inverted
        //for (int i=0;i<4;++i) size += *(pointer-4+(i^7)) << (i*8);
        size = *reinterpret_cast<const uint32_t* >(pointer);
        LogTrace("Phase2TrackerFEDBuffer") << "Condition Data size = " << size << std::endl;
        pointer+=8;
        // now the conditions
        while(pointer < stop)
        {
          // somehow the data is not inverted
          uint32_t data = 0;
          //for (int i = 0, j=3 ; i<4; i++,j--)
          //{ data += (*(pointer+i) << j*8); }
          data = *reinterpret_cast<const uint32_t*>(pointer);
          pointer += 4;

          uint32_t key  = 0;
          for (int i = 0, j=3 ; i<4; i++,j--)
          { key += (*(pointer+i) << j*8); }
          pointer += 4;

          cdata[key] = data;
        }
        // final check: cdata size == size
        if(cdata.size()!=size) {
          std::ostringstream ss;
          ss << "[Phase2Tracker::Phase2TrackerFEDBuffer::"<<__func__<<"] " << "\n";
          ss << "Number of condition data does not match the announced value!"<< "\n";
          ss << "Expected condition data Size " << size << " entries" << "\n";
          ss << "Computed condition data Size " << cdata.size() << " entries" << "\n";
          throw cms::Exception("Phase2TrackerFEDBuffer") << ss.str();
        }
      }
      // REMOVE THIS : inject fake cond data for tests
      /*
      cdata[0x0011] = 0x0001;
      cdata[0x0012] = 0x0002;
      */
      // add trigger data
      cdata[0x0B0000FF] = (TRIGGER_SIZE>0) ? (*triggerPointer_) : 0x00000000;
      return cdata;

  }
  
  FEDReadoutMode Phase2TrackerFEDBuffer::readoutMode() const
  {
    return trackerHeader_.getReadoutMode();
  }

}  // end of Phase2Tracker namespace
