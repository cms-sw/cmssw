#ifndef EventFilter_Phase2TrackerRawToDigi_Phase2TrackerPhase2TrackerFEDZSChannelUnpacker_H // {
#define EventFilter_Phase2TrackerRawToDigi_Phase2TrackerPhase2TrackerFEDZSChannelUnpacker_H

#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDDAQHeader.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDDAQTrailer.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDChannel.h"
#include <stdint.h>

namespace Phase2Tracker {

 class Phase2TrackerFEDZSChannelUnpacker
  {
  public:
    Phase2TrackerFEDZSChannelUnpacker(const Phase2TrackerFEDChannel& channel);
    virtual uint8_t clusterIndex() const;
    virtual uint8_t clusterSize() const;
    bool hasData() const { return clustersLeft_; }
    Phase2TrackerFEDZSChannelUnpacker& operator ++ ();
    Phase2TrackerFEDZSChannelUnpacker& operator ++ (int);
  protected:
    const uint8_t* data_;
    uint16_t currentOffset_; // caution : this is in bits, not bytes
    uint16_t clustersLeft_;
    uint8_t clusterdatasize_;
  };

  // unpacker for ZS CBC data
  inline Phase2TrackerFEDZSChannelUnpacker::Phase2TrackerFEDZSChannelUnpacker(const Phase2TrackerFEDChannel& channel)
    : data_(channel.data())
  {
      currentOffset_ = channel.offset() * 8 + channel.bitoffset();
  }

  inline Phase2TrackerFEDZSChannelUnpacker& Phase2TrackerFEDZSChannelUnpacker::operator ++ ()
  {
    currentOffset_ += clusterdatasize_; 
    clustersLeft_--;
    return (*this);
  }
  
  inline Phase2TrackerFEDZSChannelUnpacker& Phase2TrackerFEDZSChannelUnpacker::operator ++ (int)
  {
    ++(*this); return *this;
  }

  class Phase2TrackerFEDZSSChannelUnpacker : public Phase2TrackerFEDZSChannelUnpacker
  {
      public:
          Phase2TrackerFEDZSSChannelUnpacker(const Phase2TrackerFEDChannel& channel) : Phase2TrackerFEDZSChannelUnpacker(channel)
          {
              clusterdatasize_ = S_CLUSTER_SIZE_BITS;  
              clustersLeft_ = (channel.length()*8 - channel.bitoffset())/clusterdatasize_;
          }
          inline uint8_t clusterIndex() { return (uint8_t)read_n_at_m(data_,8,5+currentOffset_);  }
          inline uint8_t clusterSize()  { return (uint8_t)read_n_at_m(data_,3,13+currentOffset_); }
  };

  class Phase2TrackerFEDZSPChannelUnpacker : public Phase2TrackerFEDZSChannelUnpacker
  {
      public:
          Phase2TrackerFEDZSPChannelUnpacker(const Phase2TrackerFEDChannel& channel) : Phase2TrackerFEDZSChannelUnpacker(channel)
          {
              clusterdatasize_ = P_CLUSTER_SIZE_BITS;  
              clustersLeft_ = (channel.length()*8 - channel.bitoffset())/clusterdatasize_;
          }
          inline uint8_t clusterIndex() { return (uint8_t)read_n_at_m(data_,7,5+currentOffset_);  }
          inline uint8_t clusterZpos()  { return (uint8_t)read_n_at_m(data_,4,12+currentOffset_); }
          inline uint8_t clusterSize()  { return (uint8_t)read_n_at_m(data_,3,16+currentOffset_); }
  };

} // end of Phase2Tracker namespace

#endif // } end def EventFilter_Phase2TrackerRawToDigi_Phase2TrackerPhase2TrackerFEDZSChannelUnpacker_H

